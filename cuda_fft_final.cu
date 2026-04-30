#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <math_constants.h>

// Pad every 8 elements to avoid 32-bank conflicts with 16-byte elements
#define PAD(idx) ((idx) + ((idx) >> 3))

// Helper for bit reversal
__device__ int reverseBitsFinal(int x, int bits) {
    int res = 0;
    for (int i = 0; i < bits; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

__global__ void bitReverseKernelFinal(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N, int bits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int rev = reverseBitsFinal(tid, bits);
        d_out[rev] = d_in[tid];
    }
}

// Precompute twiddles linearly. Total elements: N - 1
__global__ void precomputeTwiddlesFinal(cuDoubleComplex* W, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N - 1) { 
        int p = 31 - __clz(tid + 1);
        int half_M = 1 << p;
        int k = (tid + 1) - half_M;
        int M = half_M << 1;
        double angle = -2.0 * CUDART_PI * k / M;
        double s_val, c_val;
        sincos(angle, &s_val, &c_val);
        W[tid] = make_cuDoubleComplex(c_val, s_val);
    }
}

__global__ void cudaFFTSharedTileFinal(cuDoubleComplex* d_out, const cuDoubleComplex* W, int N, int B) {
    // Partition shared memory manually into two regions
    extern __shared__ char smem[];
    int paddedElements = B + (B / 8) + 1;
    cuDoubleComplex* s_data    = (cuDoubleComplex*)smem;
    cuDoubleComplex* s_twiddle = (cuDoubleComplex*)(smem + paddedElements * sizeof(cuDoubleComplex));
    // s_twiddle needs B/2 elements max (at the last stage, half_M = B/2)
    
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * B;
    int q = B / 4;
    
    // Load input into shared memory (coalesced 4 elements per thread)
    if (blockOffset + tid < N) {
        s_data[PAD(tid)]         = d_out[blockOffset + tid];
        s_data[PAD(tid + q)]     = d_out[blockOffset + tid + q];
        s_data[PAD(tid + 2 * q)] = d_out[blockOffset + tid + 2 * q];
        s_data[PAD(tid + 3 * q)] = d_out[blockOffset + tid + 3 * q];
    }
    __syncthreads();
    
    // THREAD COARSENING (M=2 and M=4 in private registers)
    cuDoubleComplex A[4];
    A[0] = s_data[PAD(4 * tid)];
    A[1] = s_data[PAD(4 * tid + 1)];
    A[2] = s_data[PAD(4 * tid + 2)];
    A[3] = s_data[PAD(4 * tid + 3)];

    // M=2 Stage
    cuDoubleComplex t0 = A[1];
    cuDoubleComplex A1_new = cuCsub(A[0], t0);
    A[0] = cuCadd(A[0], t0);
    A[1] = A1_new;

    cuDoubleComplex t1 = A[3];
    cuDoubleComplex A3_new = cuCsub(A[2], t1);
    A[2] = cuCadd(A[2], t1);
    A[3] = A3_new;

    // M=4 Stage
    cuDoubleComplex t2 = A[2];
    cuDoubleComplex A2_new = cuCsub(A[0], t2);
    A[0] = cuCadd(A[0], t2);
    A[2] = A2_new;

    cuDoubleComplex t3 = make_cuDoubleComplex(A[3].y, -A[3].x); // W_4^1 = -i
    cuDoubleComplex A3_new2 = cuCsub(A[1], t3);
    A[1] = cuCadd(A[1], t3);
    A[3] = A3_new2;

    // Write back registers to Shared Memory
    s_data[PAD(4 * tid)]     = A[0];
    s_data[PAD(4 * tid + 1)] = A[1];
    s_data[PAD(4 * tid + 2)] = A[2];
    s_data[PAD(4 * tid + 3)] = A[3];
    __syncthreads();
    
    // WARP SYNCHRONOUS PHASE (M=8 to 64)
    // A warp has 32 threads. Here, each warp processes isolated 64-element blocks.
    // Data never crosses into other warps, so we replace block barriers with __syncwarp()
    for (int M = 8; M <= 64; M *= 2) {
        int half_M = M / 2;
        int w_offset = half_M - 1;
        
        for (int step = 0; step < 2; step++) {
            int vtid = tid + step * q;
            int group  = vtid / half_M;
            int k      = vtid % half_M;
            int even_idx = group * M + k;
            int odd_idx  = even_idx + half_M;
            
            cuDoubleComplex a = s_data[PAD(even_idx)];
            cuDoubleComplex b = s_data[PAD(odd_idx)];
            
            // Read twiddle directly from global memory (broadcast instantly via L1 cache)
            cuDoubleComplex w = W[w_offset + k];
            cuDoubleComplex t = cuCmul(w, b);
            
            s_data[PAD(even_idx)] = cuCadd(a, t);
            s_data[PAD(odd_idx)]  = cuCsub(a, t);
        }
        __syncwarp();
    }
    __syncthreads(); // Block barrier required before butterfly crosses warp boundaries
    
    // BLOCK SYNCHRONOUS PHASE (M=128 to B)
    for (int M = 128; M <= B; M *= 2) {
        int half_M = M / 2;
        int w_offset = half_M - 1;
        
        // Co-operatively load twiddles for this stage
        for (int i = tid; i < half_M; i += blockDim.x) {
            s_twiddle[i] = W[w_offset + i];
        }
        __syncthreads(); // wait for twiddle load before butterfly
        
        // Each thread processes 2 butterflies (4 elements)
        for (int step = 0; step < 2; step++) {
            int vtid = tid + step * q;
            int group  = vtid / half_M;
            int k      = vtid % half_M;
            int even_idx = group * M + k;
            int odd_idx  = even_idx + half_M;
            
            cuDoubleComplex a = s_data[PAD(even_idx)];
            cuDoubleComplex b = s_data[PAD(odd_idx)];
            
            cuDoubleComplex w = s_twiddle[k];
            cuDoubleComplex t = cuCmul(w, b);
            
            s_data[PAD(even_idx)] = cuCadd(a, t);
            s_data[PAD(odd_idx)]  = cuCsub(a, t);
        }
        __syncthreads();
    }
    
    // Write back to Global Memory (coalesced)
    if (blockOffset + tid < N) {
        d_out[blockOffset + tid]         = s_data[PAD(tid)];
        d_out[blockOffset + tid + q]     = s_data[PAD(tid + q)];
        d_out[blockOffset + tid + 2 * q] = s_data[PAD(tid + 2 * q)];
        d_out[blockOffset + tid + 3 * q] = s_data[PAD(tid + 3 * q)];
    }
}

__global__ void cudaFFTGlobalInPlaceFinal(cuDoubleComplex* d_out, const cuDoubleComplex* W, int N, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N / 2) {
        int half_M = M / 2;
        int group = tid / half_M;
        int k = tid % half_M;
        
        int even_idx = group * M + k;
        int odd_idx = even_idx + half_M;
        
        int w_offset = half_M - 1;
        
        cuDoubleComplex a = d_out[even_idx];
        cuDoubleComplex b = d_out[odd_idx];
        
        cuDoubleComplex w = W[w_offset + k];
        cuDoubleComplex t = cuCmul(w, b);
        
        d_out[even_idx] = cuCadd(a, t);
        d_out[odd_idx]  = cuCsub(a, t);
    }
}

void plan_cuda_fft_final(int N, cuDoubleComplex** d_W) {
    cudaMalloc((void**)d_W, N * sizeof(cuDoubleComplex));
    int blockSize = 256;
    int numBlocksW = (N + blockSize - 1) / blockSize;
    precomputeTwiddlesFinal<<<numBlocksW, blockSize>>>(*d_W, N);
    cudaDeviceSynchronize();
}

void destroy_cuda_fft_final(cuDoubleComplex* d_W) {
    cudaFree(d_W);
}

void run_cuda_fft_final(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const cuDoubleComplex* d_W, int N) {
    int bits = 0;
    int temp = N - 1;
    while(temp > 0) {
        bits++;
        temp >>= 1;
    }
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    bitReverseKernelFinal<<<numBlocks, blockSize>>>(d_in, d_out, N, bits);
    
    int B = 2048; 
    int tileBlocks = N / B;
    int tileThreads = B / 4;
    
    int sharedElements = B + (B / 8) + 1; // sized with padding
    int twiddleElements = B / 2;          // sizes for s_twiddle
    int sharedMemSize = (sharedElements + twiddleElements) * sizeof(cuDoubleComplex);
    
    // Some GPUs limit default dynamic shared memory to 48KB.
    // 3329 cuDoubleComplex elements * 16 bytes = 53264 bytes, which is ~52KB.
    // We explicitly request the driver to enlarge the shared memory limit.
    cudaFuncSetAttribute(cudaFFTSharedTileFinal, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    
    cudaFFTSharedTileFinal<<<tileBlocks, tileThreads, sharedMemSize>>>(d_out, d_W, N, B);
    
    for (int M = B * 2; M <= N; M *= 2) {
        int threads = N / 2;
        int grid = (threads + 255) / 256;
        cudaFFTGlobalInPlaceFinal<<<grid, 256>>>(d_out, d_W, N, M);
    }
}
