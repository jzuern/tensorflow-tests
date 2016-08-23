# Common includes for tensorflow
INCLUDES :=-I/home/jzuern/Dropbox/develop/hiwi_mrt/tensorflow/
INCLUDES +=-I/home/jzuern/Dropbox/develop/hiwi_mrt/tensorflow/bazel-tensorflow/external/eigen_archive/
INCLUDES +=-I/home/jzuern/Dropbox/develop/hiwi_mrt/tensorflow/bazel-tensorflow/external/eigen_archive/eigen-eigen-334b1d428283/Eigen/src/Core/util/

CC := g++
CCFLAGS := -std=c++11 -g
LDFLAGS :=
LDFLAGS += $(CCFLAGS)
LIBRARIES :=

# Target rules
all: build

build: unsorted_segment_sum_test


unsorted_segment_sum_test.o:unsorted_segment_sum_test.cc
	#$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<
	$(CC) $(CCFLAGS) -o $@ -c $< $(INCLUDES)

unsorted_segment_sum_test: unsorted_segment_sum_test.o
	$(CC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm unsorted_segment_sum_test.o
	rm unsorted_segment_sum_test
