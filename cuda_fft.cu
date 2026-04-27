#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <math_constants.h>

// CUDA kernel to merge sub-FFTs bottom-up
// N is the total size of the FFT
// M is the size of the merged array in the current step (M = 2, 4, 8, ..., N)
// In the python approach, we merge ffts[i] and ffts[i + half].
// half is the number of sub-arrays / 2.
// Alternatively, we view it as merging chunks.
__global__ void cudaFFTBottomUpStep(const cuDoubleComplex* curr, cuDoubleComplex* next, int N, int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Total number of elements we need to process is N / 2, because each thread can handle one butterfly (computing 2 elements).
    if (tid < N / 2) {
        int half_arrays = N / M;
        
        // Find which subarray we are in, and our local index k
        int array_idx = tid / (M / 2);
        int k = tid % (M / 2);
        
        // In the python code:
        // even_fft = ffts[i], odd_fft = ffts[i + half]
        // This means the even and odd subproblems are separated by (N / M) / 2 sub-arrays in the current level.
        // Each sub-array is M/2 in length.
        // So the distance in elements between the even and odd subproblem is (N / 2).
        
        int even_idx = array_idx * (M / 2) + k;
        int odd_idx = even_idx + N / 2;
        
        cuDoubleComplex a = curr[even_idx];
        cuDoubleComplex b = curr[odd_idx];
        
        // Twiddle factor
        double angle = -2.0 * CUDART_PI * k / M;
        double s_val, c_val;
        sincos(angle, &s_val, &c_val);
        cuDoubleComplex w = make_cuDoubleComplex(c_val, s_val);
        
        cuDoubleComplex t = cuCmul(w, b);
        
        // Output indices
        // merged array has size M, and is placed sequentially.
        int out_idx = array_idx * M + k;
        
        next[out_idx] = cuCadd(a, t);
        next[out_idx + M / 2] = cuCsub(a, t);
    }
}

void run_cuda_fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N) {
    size_t size = N * sizeof(cuDoubleComplex);

    // Allocate device memory 
    cuDoubleComplex *d_curr, *d_next;
    cudaMalloc((void**)&d_curr, size);
    cudaMalloc((void**)&d_next, size);

    cudaMemcpy(d_curr, d_in, size, cudaMemcpyDeviceToDevice);

    int blockSize = 128;
    // We launch N/2 threads in total per step
    int numThreads = N / 2;
    int gridSize = (numThreads + blockSize - 1) / blockSize;

    // Iterate through sizes M = 2, 4, 8, ..., N
    for (int M = 2; M <= N; M *= 2) {
        cudaFFTBottomUpStep<<<gridSize, blockSize>>>(d_curr, d_next, N, M);
        
        // Swap pointers for the next iteration
        cuDoubleComplex* temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }
    cudaDeviceSynchronize();

    // The result is in d_curr 
    cudaMemcpy(d_out, d_curr, size, cudaMemcpyDeviceToDevice);

    cudaFree(d_curr);
    cudaFree(d_next);
}

// Device function to reverse bits
__device__ int reverseBits(int x, int bits) {
    int res = 0;
    for (int i = 0; i < bits; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

// Global kernel for bit-reversal permutation 
__global__ void cudaFFTInPlaceShared(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N, int bits) {
    extern __shared__ cuDoubleComplex s_data[]; 
    int tid = threadIdx.x;
    if (tid < N / 2) {
        int idx1 = tid;
        int idx2 = tid + (N / 2);
        int rev1 = reverseBits(idx1, bits);
        int rev2 = reverseBits(idx2, bits);
        s_data[rev1] = d_in[idx1];
        s_data[rev2] = d_in[idx2];
    }
    __syncthreads();
    for (int M = 2; M <= N; M *= 2) {
        if (tid < N / 2) {
            int step = M / 2;
            int group = tid / step;
            int k = tid % step;
            int even_idx = group * M + k;
            int odd_idx = even_idx + step;
            cuDoubleComplex a = s_data[even_idx];
            cuDoubleComplex b = s_data[odd_idx];
            double angle = -2.0 * CUDART_PI * k / M;
            double s_val, c_val;
            sincos(angle, &s_val, &c_val);
            cuDoubleComplex w = make_cuDoubleComplex(c_val, s_val);
            cuDoubleComplex t = cuCmul(w, b);
            s_data[even_idx] = cuCadd(a, t);
            s_data[odd_idx]  = cuCsub(a, t);
        }
        __syncthreads(); 
    }
    if (tid < N / 2) {
        d_out[tid] = s_data[tid];
        d_out[tid + (N / 2)] = s_data[tid + (N / 2)];
    }
}
__global__ void bitReverseKernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N, int bits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int rev = reverseBits(tid, bits);
        d_out[rev] = d_in[tid];
    }
}

#define COARSE 4
// Tiled shared memory kernel: processes chunks of size B 
__global__ void cudaFFTSharedTile(cuDoubleComplex* d_out, int N, int B) {
    extern __shared__ cuDoubleComplex s_data[]; 
    
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * B;
    
    // Load into shared memory with Thread Coarsening
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int local_tid = tid + c * blockDim.x;
        if (blockOffset + local_tid < N) {
            s_data[local_tid] = d_out[blockOffset + local_tid];
            s_data[local_tid + B/2] = d_out[blockOffset + local_tid + B/2];
        }
    }
    __syncthreads();
    
    // Cooley-Tukey Bottom-Up
    for (int M = 2; M <= B; M *= 2) {
        int half_M = M / 2;
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int local_tid = tid + c * blockDim.x;
            if (local_tid < B / 2) {
                int group = local_tid / half_M;
                int k = local_tid % half_M;
                int even_idx = group * M + k;
                int odd_idx = even_idx + half_M;
                
                cuDoubleComplex a = s_data[even_idx];
                cuDoubleComplex b = s_data[odd_idx];
                
                double angle = -2.0 * CUDART_PI * k / M;
                double s_val, c_val;
                sincos(angle, &s_val, &c_val);
                cuDoubleComplex w = make_cuDoubleComplex(c_val, s_val);
                
                cuDoubleComplex t = cuCmul(w, b);
                
                s_data[even_idx] = cuCadd(a, t);
                s_data[odd_idx]  = cuCsub(a, t);
            }
        }
        __syncthreads(); 
    }
    
    // Write back to Global Memory

    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int local_tid = tid + c * blockDim.x;
        if (blockOffset + local_tid < N) {
            d_out[blockOffset + local_tid] = s_data[local_tid];
            d_out[blockOffset + local_tid + B/2] = s_data[local_tid + B/2];
        }
    }
}

// Global memory kernel for merge steps where M > B
__global__ void cudaFFTGlobalInPlace(cuDoubleComplex* d_out, int N, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N / 2) {
        int half_M = M / 2;
        int group = tid / half_M;
        int k = tid % half_M;
        
        int even_idx = group * M + k;
        int odd_idx = even_idx + half_M;
        
        cuDoubleComplex a = d_out[even_idx];
        cuDoubleComplex b = d_out[odd_idx];
        
        double angle = -2.0 * CUDART_PI * k / M;
        double s_val, c_val;
        sincos(angle, &s_val, &c_val);
        cuDoubleComplex w = make_cuDoubleComplex(c_val, s_val);
        
        cuDoubleComplex t = cuCmul(w, b);
        
        d_out[even_idx] = cuCadd(a, t);
        d_out[odd_idx]  = cuCsub(a, t);
    }
}

// ==========================================
// Precomputed Twiddle Factors Kernels
// ==========================================

__global__ void precomputeTwiddles(cuDoubleComplex* W, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N / 2) {
        double angle = -2.0 * CUDART_PI * k / N;
        double s_val, c_val;
        sincos(angle, &s_val, &c_val);
        W[k] = make_cuDoubleComplex(c_val, s_val);
    }
}

__global__ void cudaFFTSharedTilePrecomputed(cuDoubleComplex* d_out, const cuDoubleComplex* W, int N, int B) {
    extern __shared__ cuDoubleComplex s_data[]; 
    
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * B;
    
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int local_tid = tid + c * blockDim.x;
        if (blockOffset + local_tid < N) {
            s_data[local_tid] = d_out[blockOffset + local_tid];
            s_data[local_tid + B/2] = d_out[blockOffset + local_tid + B/2];
        }
    }
    __syncthreads();
    
    for (int M = 2; M <= B; M *= 2) {
        int half_M = M / 2;
        int step = N / M;
        
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int local_tid = tid + c * blockDim.x;
            if (local_tid < B / 2) {
                int group = local_tid / half_M;
                int k = local_tid % half_M;
                int even_idx = group * M + k;
                int odd_idx = even_idx + half_M;
                
                cuDoubleComplex a = s_data[even_idx];
                cuDoubleComplex b = s_data[odd_idx];
                
                cuDoubleComplex w = W[k * step];
                cuDoubleComplex t = cuCmul(w, b);
                
                s_data[even_idx] = cuCadd(a, t);
                s_data[odd_idx]  = cuCsub(a, t);
            }
        }
        __syncthreads(); 
    }
    
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int local_tid = tid + c * blockDim.x;
        if (blockOffset + local_tid < N) {
            d_out[blockOffset + local_tid] = s_data[local_tid];
            d_out[blockOffset + local_tid + B/2] = s_data[local_tid + B/2];
        }
    }
}

__global__ void cudaFFTGlobalInPlacePrecomputed(cuDoubleComplex* d_out, const cuDoubleComplex* W, int N, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N / 2) {
        int half_M = M / 2;
        int group = tid / half_M;
        int k = tid % half_M;
        
        int even_idx = group * M + k;
        int odd_idx = even_idx + half_M;
        
        cuDoubleComplex a = d_out[even_idx];
        cuDoubleComplex b = d_out[odd_idx];
        
        int step = N / M;
        cuDoubleComplex w = W[k * step];
        cuDoubleComplex t = cuCmul(w, b);
        
        d_out[even_idx] = cuCadd(a, t);
        d_out[odd_idx]  = cuCsub(a, t);
    }
}

void run_cuda_fft_inplace(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N) {
    int bits = 0;
    int temp = N - 1;
    while(temp > 0) {
        bits++;
        temp >>= 1;
    }
    
   
    // Step 1: Bit-reversal into d_out
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    bitReverseKernel<<<numBlocks, blockSize>>>(d_in, d_out, N, bits);
    
    // Step 2: Shared memory tiles for small M
    int B = 2048; // Max block size
    int tileBlocks = N / B;
    int tileThreads = (B / 2) / COARSE;
    if (tileThreads == 0) tileThreads = 1;
    int sharedMemSize = B * sizeof(cuDoubleComplex);
    cudaFFTSharedTile<<<tileBlocks, tileThreads, sharedMemSize>>>(d_out, N, B);
    
    // Step 3: Global memory passes for M > B
    for (int M = B * 2; M <= N; M *= 2) {
        int threads = N / 2;
        int grid = (threads + 255) / 256;
        cudaFFTGlobalInPlace<<<grid, 256>>>(d_out, N, M);
        // Kernels on the default stream execute sequentially, so no DeviceSynchronize loop needed!
    }
    cudaDeviceSynchronize();
}

void run_cuda_fft_precomputed(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N) {
    int bits = 0;
    int temp = N - 1;
    while(temp > 0) {
        bits++;
        temp >>= 1;
    }
    
    // Precompute twiddle factors on GPU
    cuDoubleComplex* d_W;
    cudaMalloc((void**)&d_W, (N / 2) * sizeof(cuDoubleComplex));
    int blockSize = 256;
    int numBlocksW = ((N / 2) + blockSize - 1) / blockSize;
    precomputeTwiddles<<<numBlocksW, blockSize>>>(d_W, N);
    
    // Step 1: Bit-reversal into d_out
    int numBlocks = (N + blockSize - 1) / blockSize;
    bitReverseKernel<<<numBlocks, blockSize>>>(d_in, d_out, N, bits);
    
    // Step 2: Shared memory tiles for small M
    int B = 2048; // Max block size
    int tileBlocks = N / B;
    int tileThreads = (B / 2) / COARSE;
    if (tileThreads == 0) tileThreads = 1;
    int sharedMemSize = B * sizeof(cuDoubleComplex);
    cudaFFTSharedTilePrecomputed<<<tileBlocks, tileThreads, sharedMemSize>>>(d_out, d_W, N, B);
    
    // Step 3: Global memory passes for M > B
    for (int M = B * 2; M <= N; M *= 2) {
        int threads = N / 2;
        int grid = (threads + 255) / 256;
        cudaFFTGlobalInPlacePrecomputed<<<grid, 256>>>(d_out, d_W, N, M);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_W);
}
