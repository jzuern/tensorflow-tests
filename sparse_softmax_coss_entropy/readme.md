# Note on python files of tensorflow installation

The tensorflow installation is in /home/jzuern/tf_installation/tensorflow/

However, the python files being executed when doing a tf graph lie in:

/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops

These files need to be changed in order implement something new


## call stack sparse_softmax_cross_entropy_with_logits


in /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py 

  def sparse_softmax_cross_entropy_with_logits(logits, labels, name=None):
    
    calls gen_nn_ops._sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)


- this source file gen_nn_ops.py does never exist in tensorflow/tensorflow/core/kernels/nn_ops.py but in
/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_nn_ops.py

  print ("hello from _sparse_softmax_cross_entropy_with_logits\n")

  result = _op_def_lib.apply_op("SparseSoftmaxCrossEntropyWithLogits",
                                features=features, labels=labels, name=name)
  return _SparseSoftmaxCrossEntropyWithLogitsOutput._make(result)


- This brings us to C++ source code where the Op is actually implemented:
/home/jzuern/tf_installation/tensorflow/tensorflow/core/kernels/sparse_xent_op.cc

template <typename Device, typename T, typename Index>
class SparseSoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
  ... blablba


      if (logits.dim_size(0) > 0) {
      functor::SparseXentFunctor<Device, T, Index> functor;
      functor(context->eigen_device<Device>(), logits.matrix<T>(),
              labels.vec<Index>(), scratch.vec<T>(), loss_out->vec<T>(),
              back_out->matrix<T>());
    }
}

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<CPUDevice, T, Index> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<CPUDevice, T, Index>::Compute(d, logits, labels,
                                                      scratch, loss, backprop);
  }
};
}  // namespace functor


SparseXentEigenImpl and all the other stuff we need for the actual calculation is defined in sparse_xent_op.h in the same source directory



## call stack weighted_cross_entropy_with_logits


in /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn.py 


Note: weighted_cross_entropy_with_logits(...) does not have a C++ OpKernel, its directly implemented as python code

def weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None):


		```
		  The usual cross-entropy cost is defined as:

		    targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))

		  The argument `pos_weight` is used as a multiplier for the positive targets:

		    targets * -log(sigmoid(logits)) * pos_weight +
			(1 - targets) * -log(1 - sigmoid(logits))

		  For brevity, let `x = logits`, `z = targets`, `q = pos_weight`.
		  The loss is:

			qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
		      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
		      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
		      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
		      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
		      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

		  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
		  the implementation uses

		      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

		```



with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and targets must have the same shape (%s vs %s)"
                       % (logits.get_shape(), targets.get_shape()))


    log_weight = 1 + (pos_weight - 1) * targets
    return math_ops.add(
        (1 - targets) * logits,
        log_weight * (math_ops.log(1 + math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),
        name=name)




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

- Shapefunction has to be commented out in user Kernel cc file because building wont work if not commented out
- weights To32Bitconst instead of To32Bit ? (sparse_xent_op.h in line 353)
- test if everything works as intended

