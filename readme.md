

# Notes

- im Makefile immer gcc-4.8 als compiler angeben (Kompabilität)
- habe Grafikkarte GT 755M
- cuda compute capability 3.0 (-arch=sm_20)

- Musste bei bazel makefile änderungen vornehmen ( siehe hier: https://github.com/tensorflow/tensorflow/issues/1066#issuecomment-200580370)
 

## TensorFLow usage


### To build with GPU support:
```
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package #run this every time build rules, c++ file or python file change)
```


### Startup of tensorflow
- run in bash in order to let TF find cuda libraries:
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
	export CUDA_HOME=/usr/local/cuda


### Compile Tensorflow unit tests with bazel:

run from workspace (tf_installation)
```
bazel build -c opt --config=cuda //tensorflow/core/kernels:segment_reduction_ops_test
```

### Run those unit tests:
```
bazel run //tensorflow/core/kernels:segment_reduction_ops_test --test_output=all --cache_test_results=no -- --benchmarks=all



```
### Clear everything compiled by bazel

```
bazel clean --expunge
```


## Versions

cuda sdk: 	7.5
cudnn: 		4.0.7
tensorflow: 	1.0
python: 	2.7.12



## Paths

tensorflow	/home/jzuern/tf_installation/tensorflow



eigenv3         /home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive

eigen_tensor	/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor

tensorflow_ops	/home/jzuern/tf_installation/tensorflow/bazel-tensorflow/tensorflow/core/kernels


cuda: 		/usr/local/cuda-7.5

cudnn: 		/usr/local/cuda-7.5

python: 	/usr/bin/python

bazel: 		/home/jzuern/bin/bazel


# Definitions

## CUDA stream object

A stream is a queue of device work (kernel launches, memory copies can be launched from within a stream). Host places work in queue und continues immediately (asynchronous). Declaration of stream handle: cudaStream_t stream; cudaStreamCreate(&stream); (allocates a stream).
Placing work into stream: 4th launch parameter: kernel <<<blocks,threads,smem,stream>>>();
Placed in some API calls as well: cudaMemcpyAsync(dst,src,size,dir,stream);






