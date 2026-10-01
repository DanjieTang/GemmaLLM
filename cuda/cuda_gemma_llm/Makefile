# Build the CUDA Gemma decoder benchmark. Override the GPU architecture with,
# for example, `make ARCH=sm_90` (the default builds for the local GPU).
NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SOURCES := main.cu model.cu kernels.cu config.cu
HEADERS := common.cuh model.cuh kernels.cuh

gemma_llm: $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -lcublas -o $@

clean:
	rm -f gemma_llm

.PHONY: clean
