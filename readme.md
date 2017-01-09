
## Versions

cuda sdk: 	7.5
cudnn: 		4.0.7
tensorflow: 	0.12
python: 	2.7.12

CUDA compute capability = 3.0 (-arch=sm_20)

## Hardware 
GPU: GT 755M


## Paths

tensorflow	/home/jzuern/tensorflow
eigenv3         /home/jzuern/tensorflow/bazel-tensorflow/external/eigen_archive
eigen_tensor	/home/jzuern/tensorflow/bazel-tensorflow/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor
tensorflow_ops	/home/jzuern/tensorflow/bazel-tensorflow/tensorflow/core/kernels
cuda: 		/usr/local/cuda-7.5
cudnn: 		/usr/local/cuda-7.5
python: 	/usr/bin/python
bazel: 		/home/jzuern/bin/bazel



## TensorFlow


### To build TF with GPU support:
```
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package #run this every time build rules, c++ file or python file change)
```


### Startup of tensorflow
- run in bash in order to let TF find cuda libraries:
```
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
$ export CUDA_HOME=/usr/local/cuda
```
- then run python name_of_python_script.py
(or add those commands to your bashrc


### Compile Tensorflow unit tests with bazel:

run from workspace (tf_installation):
```
bazel build -c opt --config=cuda //tensorflow/core/kernels:segment_reduction_ops_test
```

### Run those unit tests:
```
bazel run //tensorflow/core/kernels:segment_reduction_ops_test --test_output=all --cache_test_results=no -- --benchmarks=all

```


### Compile shared library with bazel (dont forget '--config=cuda' !)
```
$ bazel build -c opt --config=cuda //tensorflow/core/user_ops:zero_out.so
```
This will place the .so file in directory /tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so

### Compile shared library withOUT - CPU
```
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64/"
$ export CUDA_HOME=/usr/local/cuda-7.5
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -I /home/jzuern/tf_installation/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0
```

### Compile shared library withOUT bazel - GPU
```
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64/"
$ export CUDA_HOME=/usr/local/cuda-7.5
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ /usr/local/cuda-7.5/bin/nvcc -ccbin g++-4.8 -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
$ g++-4.8 -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc cuda_op_kernel.cu.o -I $TF_INC -fPIC -lcudart -L/usr/local/cuda/lib64/
```



## Wichtiges Q&A

### wie sage ich Bazel, dass .h Dateien als Header eingebunden werden?
Lösung: in tensorflow/tensorflow.bzl die Definition der Funktion tf_custom_op_library erweitern um hdr Parameter. Übergebe es internal.cc_binary(...) wenn gpu_src defined ist


### Other
- im Makefile immer gcc-4.8 als compiler angeben (Kompabilität)
- Musste bei bazel makefile änderungen vornehmen ( siehe hier: https://github.com/tensorflow/tensorflow/issues/1066#issuecomment-200580370)
 





