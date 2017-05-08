### Compile shared library:
$ cd /home/jzuern/tensorflow/tensorflow/core/user_ops
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared dilated_maxpooling.cc -o dilated_maxpooling.so -fPIC -I $TF_INC -I /home/jzuern/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0




### Compile shared library with bazel
```
$ bazel build -c opt --config=opt //tensorflow/core/user_ops:zero_out.so
```
This will place the .so file in directory /tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops
