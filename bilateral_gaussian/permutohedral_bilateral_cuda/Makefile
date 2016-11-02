CUDA_PATH ?= /usr/local/cuda-7.5
CC := g++-4.8
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CC)


bilateral: Image.cpp Image.h bilateral.cpp jpg_file.h png_file.h macros.h permutohedral.o
	$(NVCC) bilateral.cpp Image.cpp permutohedral.o -o bilateral  -lpng -ljpeg -L. -lcutil

permutohedral.o: permutohedral.cu hash_table.cu
	# $(NVCC) -arch=sm_20 -O3 -use_fast_math -DLIBRARY -c permutohedral.cu -o permutohedral.o
	$(NVCC) -arch=sm_20 -DLIBRARY -c permutohedral.cu -o permutohedral.o

clean:
	rm bilateral permutohedral.o
