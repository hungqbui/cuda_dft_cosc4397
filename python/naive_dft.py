import cmath
import math

def dft_1d(x):
    N = len(x)
    X = [0j] * N

    
    for k in range(N):
        s = 0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += x[n] * cmath.exp(angle)
        X[k] = s

    return X


if __name__ == "__main__":
    x = [1, 2, 3, 4]
    X = dft_1d(x)


    for i, val in enumerate(X):
        print(f"k={i}: {val}")

