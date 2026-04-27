NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo
LIBS = -lcufft

CUDA_SRCS = cuda_dftv1.cu cuda_fft.cu main.cu
OBJS = $(CUDA_SRCS:.cu=.o)

TARGET = compare_ffts

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
