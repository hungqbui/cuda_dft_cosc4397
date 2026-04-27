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
    
    // Load input into shared memory (unchanged)
    if (blockOffset + tid < N && tid < B / 2) {
        s_data[PAD(tid)]       = d_out[blockOffset + tid];
        s_data[PAD(tid + B/2)] = d_out[blockOffset + tid + B/2];
    }
    __syncthreads();
    
    for (int M = 2; M <= B; M *= 2) {
        int half_M = M / 2;
        int w_offset = half_M - 1;
        
        // Cooperatively load this stage's twiddle slice into shared memory.
        // Only half_M values are needed. All threads participate to keep it fast.
        // At early stages half_M is tiny (1, 2, 4...) so most threads idle here,
        // but the coalescing benefit at those stages is highest so it's worth it.
        if (tid < half_M) {
            s_twiddle[tid] = W[w_offset + tid];
        }
        __syncthreads(); // wait for twiddle load before butterfly
        
        if (tid < B / 2) {
            int group  = tid / half_M;
            int k      = tid % half_M;
            int even_idx = group * M + k;
            int odd_idx  = even_idx + half_M;
            
            cuDoubleComplex a = s_data[PAD(even_idx)];
            cuDoubleComplex b = s_data[PAD(odd_idx)];
            
            cuDoubleComplex w = s_twiddle[k]; // shared memory read instead of global W
            cuDoubleComplex t = cuCmul(w, b);
            
            s_data[PAD(even_idx)] = cuCadd(a, t);
            s_data[PAD(odd_idx)]  = cuCsub(a, t);
        }
        __syncthreads();
    }
    
    // Write back (unchanged)
    if (blockOffset + tid < N && tid < B / 2) {
        d_out[blockOffset + tid]       = s_data[PAD(tid)];
        d_out[blockOffset + tid + B/2] = s_data[PAD(tid + B/2)];
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
    int tileThreads = B / 2;
    
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
