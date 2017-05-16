### Compile shared library:
$ cd /home/jzuern/tensorflow/tensorflow/core/user_ops
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared dilated_maxpooling.cc -o dilated_maxpooling.so -fPIC -I $TF_INC -I /home/jzuern/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0




### Compile shared library with bazel
```
$ bazel build -c opt --config=opt //tensorflow/core/user_ops:zero_out.so
```
This will place the .so file in directory /tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops




## python OP: Baue tensorflow/tensorflow/pyton/ops/nn_ops (hier ist max_pool definiert)
```
$ bazel build --config opt //tensorflow/python:nn_ops
```

- Das Problem in der verlinkten Funktion ist hauptsächlich, dass dilation_rate und stride nicht gleichzeitig Werte > 1 haben können.



## Doch C++ Custom Op:
- Brauchen kein Padding, ...
- GPU Implementierung wichtig


# disable gpu usage: export CUDA_VISIBLE_DEVICES=-1


# about Padding:

Padding: the padding we apply to the input tensor along the rows and columns dimensions. This is usually used to make sure that the spatial dimensions do not shrink when we progress with convolutions. Two types of padding are supported:
VALID: No padding is carried out.
SAME: The pad value is computed so that the output will have the same dimensions as the input.
The padded area is zero-filled.

enum Padding {
  VALID = 1,  // No padding.
  SAME = 2,   // Input and output layers have the same size.
};
