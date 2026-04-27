#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#include <math_constants.h>

__global__ void cudaDFT(const cuDoubleComplex* x, cuDoubleComplex* y, int N) {

    int k = (threadIdx.x + blockDim.x * blockIdx.x) * 2; // each thread compute 2 values

    cuDoubleComplex s1 = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex s2 = make_cuDoubleComplex(0.0, 0.0);

    __shared__ cuDoubleComplex tile[128]; // Match blockSize in main

    for (int j = 0; j < (N + blockDim.x - 1) / blockDim.x; j++) {
        int start = j * blockDim.x;
        
        // each thread loads one element of the tile
        if (start + threadIdx.x < N) {
            tile[threadIdx.x] = x[start + threadIdx.x];
        } else {
            tile[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
        }
        __syncthreads();

        // compute partial results over the shared tile
        int limit = (start + blockDim.x < N) ? blockDim.x : (N - start);
        for (int i = 0; i < limit; i++) {
            int n = start + i;
            cuDoubleComplex cur = tile[i];
            
            if (k < N) {
                double angle1 = -2.0 * CUDART_PI * ((k * n) % N) / (float)N;
                double s_val1, c_val1;
                sincos(angle1, &s_val1, &c_val1);
                cuDoubleComplex w1 = make_cuDoubleComplex(c_val1, s_val1);
                s1 = cuCadd(s1, cuCmul(cur, w1));
            }
            if (k + 1 < N) {
                double angle2 = -2.0 * CUDART_PI * (((k+1) * n) % N) / (float)N;
                double s_val2, c_val2;
                sincos(angle2, &s_val2, &c_val2);
                cuDoubleComplex w2 = make_cuDoubleComplex(c_val2, s_val2);
                s2 = cuCadd(s2, cuCmul(cur, w2));
            }
        }
        __syncthreads(); 
    }

    // coarsened write

    if (k < N) y[k] = s1;
    if (k + 1 < N) y[k+1] = s2;
}

void run_cuda_dft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N) {
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaDFT<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
}