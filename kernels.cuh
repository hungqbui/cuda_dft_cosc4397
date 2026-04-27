#pragma once
#include <cuComplex.h>

void run_cuda_dft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft_inplace(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft_precomputed(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void plan_cuda_fft_final(int N, cuDoubleComplex** d_W);
void destroy_cuda_fft_final(cuDoubleComplex* d_W);
void run_cuda_fft_final(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const cuDoubleComplex* d_W, int N);
