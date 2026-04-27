#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>
#include "kernels.cuh"

// Host Out-of-place (Bottom-Up) FFT
void host_fft_bottomup(const cuDoubleComplex* in, cuDoubleComplex* out, int N) {
    cuDoubleComplex* curr = new cuDoubleComplex[N];
    cuDoubleComplex* next = new cuDoubleComplex[N];
    memcpy(curr, in, N * sizeof(cuDoubleComplex));

    for (int M = 2; M <= N; M *= 2) {
        for (int tid = 0; tid < N / 2; tid++) {
            int array_idx = tid / (M / 2);
            int k = tid % (M / 2);
            
            int even_idx = array_idx * (M / 2) + k;
            int odd_idx = even_idx + N / 2;
            
            cuDoubleComplex a = curr[even_idx];
            cuDoubleComplex b = curr[odd_idx];
            
            double angle = -2.0 * CUDART_PI * k / M;
            cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
            cuDoubleComplex t = cuCmul(w, b);
            
            int out_idx = array_idx * M + k;
            next[out_idx] = cuCadd(a, t);
            next[out_idx + M / 2] = cuCsub(a, t);
        }
        cuDoubleComplex* temp = curr;
        curr = next;
        next = temp;
    }
    memcpy(out, curr, N * sizeof(cuDoubleComplex));
    delete[] curr;
    delete[] next;
}

// Host In-place CPU FFT bit reversal helper
int host_reverseBits(int x, int bits) {
    int res = 0;
    for (int i = 0; i < bits; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

// Host In-place (Bit-Reversed) FFT
void host_fft_inplace(const cuDoubleComplex* in, cuDoubleComplex* out, int N) {
    memcpy(out, in, N * sizeof(cuDoubleComplex));
    int bits = 0;
    int temp = N - 1;
    while(temp > 0) { bits++; temp >>= 1; }

    for(int i=0; i<N; i++) {
        int j = host_reverseBits(i, bits);
        if (i < j) {
            cuDoubleComplex tmp = out[i];
            out[i] = out[j];
            out[j] = tmp;
        }
    }

    for(int M=2; M<=N; M*=2) {
        int half = M / 2;
        
        // Loop over j first! This lets us only calculate sin/cos (M/2) times per stage
        // rather than (N/2) times per stage, dropping overall trig calls from N log N down to just N.
        for(int j=0; j<half; j++) {
            double angle = -2.0 * CUDART_PI * j / M;
            cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
            
            for(int k=0; k<N; k+=M) {
                cuDoubleComplex even = out[k+j];
                cuDoubleComplex odd = out[k+j+half];
                cuDoubleComplex t = cuCmul(w, odd);
                out[k+j] = cuCadd(even, t);
                out[k+j+half] = cuCsub(even, t);
            }
        }
    }
}

int main(int argc, char** argv) {
    int p = 11; // Default to 2^11 = 2048
    if (argc > 1) {
        p = atoi(argv[1]);
    }
    
    if (p < 1 || p > 25) { // Protect against overflow and out-of-memory
        printf("Error: Please provide a power (exponent) between 1 and 25.\n");
        return 1;
    }
    
    int N = 1 << p;
    
    size_t size = N * sizeof(cuDoubleComplex);

    // Allocate host memory
    cuDoubleComplex *h_x = new cuDoubleComplex[N];
    cuDoubleComplex *h_dft_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_fft_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_fft_inplace_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_fft_precomp_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_fft_final_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_cufft_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_host_bottomup_y = new cuDoubleComplex[N];
    cuDoubleComplex *h_host_inplace_y = new cuDoubleComplex[N];

    // Initialize input with values 1 to N (Real part only)
    for (int i = 0; i < N; i++) {
        h_x[i] = make_cuDoubleComplex((float)(i + 1), 0.0);
    }

    // Allocate device memory
    cuDoubleComplex *d_x, *d_dft_y, *d_fft_y, *d_fft_inplace_y, *d_fft_precomp_y, *d_fft_final_y, *d_cufft_y;
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_dft_y, size);
    cudaMalloc((void**)&d_fft_y, size);
    cudaMalloc((void**)&d_fft_inplace_y, size);
    cudaMalloc((void**)&d_fft_precomp_y, size);
    cudaMalloc((void**)&d_fft_final_y, size);
    cudaMalloc((void**)&d_cufft_y, size);

    // Copy input to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Setup CUDA events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float s_dft = 0, s_fft = 0, s_fft_inplace = 0, s_fft_precomp = 0, s_fft_final= 0, s_cufft = 0;
    float s_host_bu = 0, s_host_ip = 0;

    // ==========================================
    // 1. Naive O(N^2) DFT
    // ==========================================
    if (p < 18) {
        cudaEventRecord(start);
        run_cuda_dft(d_x, d_dft_y, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&s_dft, start, stop);
    }

    // ==========================================
    // 2. Custom O(N log N) FFT
    // ==========================================
    cudaEventRecord(start);
    run_cuda_fft(d_x, d_fft_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_fft, start, stop);
    
    // ==========================================
    // 3. Custom O(N log N) Shared In-Place FFT 
    // ==========================================
    cudaEventRecord(start);
    run_cuda_fft_inplace(d_x, d_fft_inplace_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_fft_inplace, start, stop);
    
    // ==========================================
    // 3.5 Custom Precomputed Shared In-Place FFT 
    // ==========================================
    cudaEventRecord(start);
    run_cuda_fft_precomputed(d_x, d_fft_precomp_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_fft_precomp, start, stop);

    // ==========================================
    // 3.8 Custom Final (Bank/Coalesced) Shared FFT
    // ==========================================
    cuDoubleComplex* d_W_final;
    plan_cuda_fft_final(N, &d_W_final);
    
    cudaEventRecord(start);
    run_cuda_fft_final(d_x, d_fft_final_y, d_W_final, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_fft_final, start, stop);
    
    destroy_cuda_fft_final(d_W_final);

    // ==========================================
    // 4. cuFFT
    // ==========================================
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cudaEventRecord(start);
    cufftExecZ2Z(plan, (cufftDoubleComplex *)d_x, (cufftDoubleComplex *)d_cufft_y, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_cufft, start, stop);

    // ==========================================
    // 5. CPU O(N log N) Out-of-Place FFT
    // ==========================================
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    host_fft_bottomup(h_x, h_host_bottomup_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_host_bu, start, stop);

    // ==========================================
    // 6. CPU O(N log N) In-Place FFT 
    // ==========================================
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    host_fft_inplace(h_x, h_host_inplace_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&s_host_ip, start, stop);

    // ==========================================
    // Copy results back and compute errors
    // ==========================================
    if (p < 18)
        cudaMemcpy(h_dft_y, d_dft_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fft_y, d_fft_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fft_inplace_y, d_fft_inplace_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fft_precomp_y, d_fft_precomp_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fft_final_y, d_fft_final_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cufft_y, d_cufft_y, size, cudaMemcpyDeviceToHost);

    double dft_max_error = 0.0;
    double fft_max_error = 0.0;
    double fft_inplace_max_error = 0.0;
    double fft_precomp_max_error = 0.0;
    double fft_final_max_error = 0.0;
    double host_bu_max_error = 0.0;
    double host_ip_max_error = 0.0;

    for (int i = 0; i < N; i++) {
        auto compute_err = [&](cuDoubleComplex a, cuDoubleComplex b) {
            double dx = a.x - b.x;
            double dy = a.y - b.y;
            return sqrt(dx*dx + dy*dy);
        };
        
        double err = compute_err(h_fft_y[i], h_cufft_y[i]);
        if(err > fft_max_error) fft_max_error = err;
        
        if (p < 18) {
            err = compute_err(h_dft_y[i], h_cufft_y[i]);
            if(err > dft_max_error) dft_max_error = err;
        }
        
        err = compute_err(h_fft_inplace_y[i], h_cufft_y[i]);
        if(err > fft_inplace_max_error) fft_inplace_max_error = err;
        
        err = compute_err(h_fft_precomp_y[i], h_cufft_y[i]);
        if(err > fft_precomp_max_error) fft_precomp_max_error = err;
        
        err = compute_err(h_fft_final_y[i], h_cufft_y[i]);
        if(err > fft_final_max_error) fft_final_max_error = err;

        err = compute_err(h_host_bottomup_y[i], h_cufft_y[i]);
        if(err > host_bu_max_error) host_bu_max_error = err;
        
        err = compute_err(h_host_inplace_y[i], h_cufft_y[i]);
        if(err > host_ip_max_error) host_ip_max_error = err;
    }

    printf("======================================\n");
    printf("Transforms for N = %d\n", N);
    printf("======================================\n");
    printf("%-26s | %-12s | %-12s\n", "Implementation", "Time (s)", "Max Error vs cuFFT");
    printf("--------------------------------------\n");
    if (p < 18)
        printf("%-26s | %-12.8f | %-12.4e\n", "GPU Naive DFT", s_dft / 1e3, dft_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "CPU Naive Merging FFT", s_host_bu / 1e3, host_bu_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "CPU In-Place FFT", s_host_ip / 1e3, host_ip_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "GPU Naive Merging FFT", s_fft / 1e3, fft_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "GPU Shared In-Place FFT", s_fft_inplace / 1e3, fft_inplace_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "GPU Shared Precomp FFT", s_fft_precomp / 1e3, fft_precomp_max_error);
    printf("%-26s | %-12.8f | %-12.4e\n", "GPU Final Optimized FFT", s_fft_final / 1e3, fft_final_max_error);
    printf("%-26s | %-12.8f | %-12s\n", "GPU NVIDIA cuFFT", s_cufft / 1e3, "0.0000e+00");
    printf("======================================\n");

    // Cleanup
    cufftDestroy(plan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_dft_y);
    cudaFree(d_fft_y);
    cudaFree(d_fft_inplace_y);
    cudaFree(d_fft_precomp_y);
    cudaFree(d_fft_final_y);
    cudaFree(d_cufft_y);
    delete[] h_x;
    delete[] h_dft_y;
    delete[] h_fft_y;
    delete[] h_fft_inplace_y;
    delete[] h_fft_precomp_y;
    delete[] h_fft_final_y;
    delete[] h_cufft_y;
    delete[] h_host_bottomup_y;
    delete[] h_host_inplace_y;

    return 0;
}
