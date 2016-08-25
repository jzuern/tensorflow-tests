# Common includes for tensorflow
INCLUDES :=-I/home/jzuern/tf_installation/tensorflow/
INCLUDES +=-I/usr/local/lib/python2.7/dist-packages/tensorflow/include/google/protobuf/stubs
INCLUDES +=-I/home/jzuern/tf_installation/tensorflow/bazel-genfiles/
INCLUDES +=-I/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive/
INCLUDES +=-I/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive/Eigen/src/Core/util/
INCLUDES +=-I/home/jzuern/tf_installation/tensorflow/bazel-genfiles/tensorflow/core/framework/


CC := g++
CCFLAGS := -std=c++11 -g
LDFLAGS :=
LDFLAGS += $(CCFLAGS)
LIBRARIES :=

# Target rules
all: build

build: unsorted_segment_sum_test


unsorted_segment_sum_test.o:unsorted_segment_sum_test.cc
	$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<
	#$(CC) $(CCFLAGS) -o $@ -c $< $(INCLUDES)

unsorted_segment_sum_test: unsorted_segment_sum_test.o
	$(CC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm unsorted_segment_sum_test.o
	rm unsorted_segment_sum_test
