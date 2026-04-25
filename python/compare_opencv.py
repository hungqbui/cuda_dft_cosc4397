import numpy as np
import cv2
import math
from naive_dft import dft_1d

def to_opencv_format(x):
    """
    Convert 1D real signal to OpenCV complex format (Nx2 array)
    """
    x = np.array(x, dtype=np.float32)
    complex_array = np.zeros((len(x), 2), dtype=np.float32)
    complex_array[:, 0] = x  # real part
    return complex_array

def from_opencv_format(X_cv):
    """
    Convert OpenCV output (Nx2) to Python complex list
    """
    return [complex(r, i) for r, i in X_cv]

def compare_dft(x):
    print("Input:", x)

  
    # Naive DFT
    X_naive = dft_1d(x)

    # OpenCV DFT
    x_cv = to_opencv_format(x)
    X_cv = cv2.dft(x_cv)
    X_cv_complex = from_opencv_format(X_cv)

    # Compare
    max_error = 0.0

    print("\nResults:")
    for i in range(len(x)):
        naive = X_naive[i]
        cv = X_cv_complex[i]

        error = abs(naive - cv)
        max_error = max(max_error, error)

        print(f"k={i}: naive={naive:.5f}, cv={cv:.5f}, error={error:.5e}")

    print("\nMax error:", max_error)


if __name__ == "__main__":
    # Test signal
    x = [1, 2, 3, 4]

    compare_dft(x)

