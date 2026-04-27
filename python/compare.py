import time
import numpy as np
import cv2
import math
from naive_dft import dft_1d
from fft import fft1d, fft1d_bottomup, fft1d_iter_bitmap
from scipy.fft import fft as scipy_fft

def to_opencv_format(x):
    """
    Convert 1D real signal to OpenCV complex format (Nx2 array)
    """
    x = np.array(x, dtype=np.float64)
    complex_array = np.zeros((len(x), 1, 2), dtype=np.float64)
    complex_array[:, 0, 0] = x  # real part
    return complex_array

def from_opencv_format(X_cv):
    """
    Convert OpenCV output (Nx2) to Python complex list
    """
    return [complex(r, i) for r, i in X_cv.reshape(-1, 2)]

def compare_dft(x):
    # print("Input:", x)

    # Naive DFT
    t0 = time.time()
    X_naive = dft_1d(x)
    naive_time = time.time() - t0

    # OpenCV DFT
    x_cv = to_opencv_format(x)
    t1 = time.time()
    X_cv = cv2.dft(x_cv)
    cv_time = time.time() - t1
    X_cv_complex = from_opencv_format(X_cv)

    # FFT
    t2 = time.time()
    X_fft = fft1d(x)
    fft_time = time.time() - t2

    t3 = time.time()
    X_fft_bottomup = fft1d_bottomup(x)
    fft_bottomup_time = time.time() - t3

    t4 = time.time()
    X_scipy = scipy_fft(x)
    scipy_time = time.time() - t4

    t5 = time.time()
    X_fft_iter_bitmap = fft1d_iter_bitmap(x)
    fft_iter_bitmap_time = time.time() - t5
    

    # Compare
    max_error1 = 0.0
    max_error2 = 0.0
    max_error3 = 0.0
    max_error4 = 0.0

    print("\nResults:")
    for i in range(len(x)):
        naive = X_naive[i]
        cv = X_cv_complex[i]
        fft = X_fft[i]
        error1 = abs(naive - cv)
        max_error1 = max(max_error1, error1)
        error2 = abs(fft - cv)
        max_error2 = max(max_error2, error2)
        error3 = abs(fft - naive)
        max_error3 = max(max_error3, error3)
        error4 = abs(X_fft_iter_bitmap[i] - cv)
        max_error4 = max(max_error4, error4)


    print("Max error Naive:", max_error1)
    print("Max error FFT:", max_error2)
    print("Max error FFT (Bottom-up):", max_error3)
    print("Max error FFT (Bottom-up Bitmap Inplace):", max_error4)
    print("\nRuntimes:")
    print(f"Naive DFT: {naive_time:.8f} seconds")
    print(f"OpenCV DFT: {cv_time:.8f} seconds")
    print(f"Scipy FFT: {scipy_time:.8f} seconds")
    print(f"FFT: {fft_time:.8f} seconds")
    print(f"FFT (Bottom-up): {fft_bottomup_time:.8f} seconds")
    print(f"FFT (Bottom-up Bitmap Inplace): {fft_iter_bitmap_time:.8f} seconds")

if __name__ == "__main__":
    # Test signal
    p = input("Enter power of 2 for input size (e.g., 10 for 1024): ")
    
    x = [i for i in range(1, int(2**int(p)+1))]
    print("N =", 2**int(p))
    compare_dft(x)

