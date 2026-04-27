#pragma once
#include <cuComplex.h>

void run_cuda_dft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft_inplace(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
void run_cuda_fft_precomputed(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);
