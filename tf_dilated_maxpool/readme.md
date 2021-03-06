### Compile shared library:
```
$ cd /home/jzuern/tensorflow/tensorflow/core/user_ops
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared dilated_maxpooling.cc -o dilated_maxpooling.so -fPIC -I $TF_INC -I /home/jzuern/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0
```

### Compile shared library with bazel
```
$ bazel build -c opt --config=opt //tensorflow/core/user_ops:zero_out.so
```
This will place the .so file in directory /tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops



## python OP: Baue tensorflow/tensorflow/pyton/ops/nn_ops (hier ist max_pool definiert)
```
$ bazel build --config opt //tensorflow/python:nn_ops
```


# C++ Custom Op with GPU CUDA:
```
$ nvcc --compiler-bindir /usr/bin/gcc-4.8 -std=c++11 -c -o dilated_maxpooling_gpu.cu.o dilated_maxpooling_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
$ g++ -std=c++11 -shared -o dilated_maxpooling_gpu.so dilated_maxpooling_gpu.cc dilated_maxpooling_gpu.cu.o -I $TF_INC -I /home/jzuern/tensorflow/ -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0
```
