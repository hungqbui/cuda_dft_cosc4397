from naive_dft import dft_1d

def test_basic():
    x = [1, 0, 0, 0]
    X = dft_1d(x)


# Expected: all ones
    for val in X:
        assert abs(val - 1) < 1e-6

print("Test passed!")


if __name__ == "__main__":
    test_basic()
