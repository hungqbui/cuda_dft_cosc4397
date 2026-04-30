import math
import cmath

def fft1d(x):
    """
    Compute the 1D FFT of a real-valued input signal x using the Cooley-Tukey algorithm.
    This implementation assumes the length of x is a power of 2.
    """
    N = len(x)
    if N <= 1:
        return list(x) # Explicitly return as list to avoid reference bugs
        
    # Strictly enforce power of 2 length
    if N & (N - 1) != 0:
        raise ValueError(f"Length of input must be a power of 2, but got {N}.")
    
    # Recursive FFT on even and odd indexed elements
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])
    
    # Combine
    X = [0] * N
    for k in range(N // 2):
        t = complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)) * odd[k]
        X[k] = even[k] + t
        X[k + N // 2] = even[k] - t
    
    return X

def fft1d_bottomup(x):
    N = len(x)
    if N <= 1:
        return list(x)
    
    # Strictly enforce power of 2 length
    if N & (N - 1) != 0:
        raise ValueError(f"Length of input must be a power of 2, but got {N}.")
    
    # Start with N arrays of length 1
    ffts = [[val] for val in x]
    
    # Iteratively merge the smaller FFT arrays into larger ones
    while len(ffts) > 1:
        next_ffts = []
        half = len(ffts) // 2
        
        for i in range(half):
            even_fft = ffts[i]
            odd_fft = ffts[i + half]
            
            M = len(even_fft) * 2
            merged = [0] * M
            
            for k in range(M // 2):
                t = cmath.exp(-2j * math.pi * k / M) * odd_fft[k]
                merged[k] = even_fft[k] + t
                merged[k + M // 2] = even_fft[k] - t
                
            next_ffts.append(merged)
        print(next_ffts)
        ffts = next_ffts
        
    # The final array is our full FFT result
    return ffts[0]

def fft1d_iter_bitmap(x):
    x = list(x)
    
    N = len(x)
    if N <= 1:
        return list(x)
    
    # Strictly enforce power of 2 length
    if N & (N - 1) != 0:
        raise ValueError(f"Length of input must be a power of 2, but got {N}.")
    
    def bit_reverse(n, bits):
        result = 0
        for i in range(bits):
            result <<= 1
            result |= (n & 1)
            n >>= 1
        return result
    
    for i in range(N):
        j = bit_reverse(i, N.bit_length() - 1)
        if i < j:
            x[i], x[j] = x[j], x[i]
            
    m = 2
    while m <= N:
        half = m // 2
        w_m = cmath.exp(-2j * math.pi / m)
        
        for k in range(0, N, m):
            w = 1
            for j in range(half):
                even = x[k + j]
                odd = x[k + j + half]
                
                x[k + j] = even + w * odd
                x[k + j + half] = even - w * odd
                w *= w_m

        m *= 2
        
    return x
            