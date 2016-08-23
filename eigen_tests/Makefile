# Common includes for Eigen
INCLUDES:= -I/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive


CC := g++
CCFLAGS := -std=c++11 -g
LDFLAGS :=
LDFLAGS += $(CCFLAGS)
LIBRARIES :=

# Target rules
all: build

build: eigen_test_tensor_reduction


eigen_test_tensor_reduction.o:eigen_test_tensor_reduction.cc
	$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

eigen_test_tensor_reduction: eigen_test_tensor_reduction.o
	$(CC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm eigen_test_tensor_reduction.o
	rm eigen_test_tensor_reduction
