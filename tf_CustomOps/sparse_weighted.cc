


#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/sparse_xent_op.h"


namespace tensorflow {


  REGISTER_OP("SparseWeighted")
      .Input("features: T")
      .Input("labels: Tlabels")
      .Input("weights: T")
      .Output("loss: T")
      .Output("backprop: T")
      .Attr("T: {half, float, double}")
      .Attr("Tlabels: {int32, int64} = DT_INT64")
      .Doc(R"doc(
  See`SparseSoftmaxCrossEntropyWithLogits` for details.

  It's like  SparseSoftmaxCrossEntropyWithLogits but with class weights for weighted loss caluclation based on ground-truth label weight.

  )doc");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class SparseWeightedOp : public OpKernel {
 public:
  explicit SparseWeightedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    printf("Hello from SparseWeightedOp Compute\n");
    const Tensor& logits = context->input(0);
    const Tensor& labels = context->input(1);
    const Tensor& weights = context->input(2); // jzuern

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits.shape()),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(weights.shape()), // jzuern
                errors::InvalidArgument("weights must be 1-D, but got shape ",
                                        labels.shape().DebugString()));


    OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits.shape().DebugString(), " and labels shape ",
                    labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits.shape().DebugString()));

    OP_REQUIRES(context, logits.dim_size(1) == weights.dim_size(0),
                errors::InvalidArgument(
                    "2nd dimension of logits and dimension of weight must have the same size, "
                    "got logits shape ", logits.shape().DebugString(),
                    " and weights shape ", weights.shape().DebugString()));

    Tensor scratch;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   labels.shape(), &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, labels.shape(), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, logits.shape(), &back_out));

    if (logits.dim_size(0) > 0) {
      functor::SparseXentFunctor_jzuern<Device, T, Index> functor;
      functor(context->eigen_device<Device>(),
              logits.matrix<T>(),
              labels.vec<Index>(),
              weights.vec<T>(), // jzuern
              scratch.vec<T>(),
              loss_out->vec<T>(),
              back_out->matrix<T>());
    }
  }
};


// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<CPUDevice, T, Index> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<CPUDevice, T, Index>::Compute(d, logits, labels,
                                                      scratch, loss, backprop);
  }
};



// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
template <typename T, typename Index>
struct SparseXentFunctor_jzuern<CPUDevice, T, Index> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::ConstVec weights,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl_jzuern<CPUDevice, T, Index>::Compute(d, logits, labels,
                                                      weights,
                                                      scratch, loss, backprop);
  }
};
} // end namespace functor





#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("SparseWeighted") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      SparseWeightedOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64)

#if GOOGLE_CUDA
REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64)
#endif  // GOOGLE_CUDA

#undef REGISTER



}  // namespace tensorflow
