# Note on python files of tensorflow installation

The tensorflow installation is in /home/jzuern/tf_installation/tensorflow/

However, the python files being executed when doing a tf graph lie in:

/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops

These files need to be changed in order implement something new




## implementing a custom Op


- Create file tensorflow/core/user_ops/zero_out.cc


(#)include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
     // implementation
  }
};





## implementing a custom Op


does not work:
$ bazel build -c opt //tensorflow/core/user_ops:zero_out.so

does work:
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared sparse_weighted.cc -o sparse_weighted.so -fPIC -I $TF_INC -I /home/jzuern/tf_installation/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0


## using the Op in python

import tensorflow as tf
zero_out_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/sparse_weighted.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()




## Problems
- ...

