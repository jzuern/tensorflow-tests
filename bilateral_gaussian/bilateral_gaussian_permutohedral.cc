
#define EIGEN_USE_THREADS



#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "permutohedral.h"


namespace tensorflow {


  REGISTER_OP("BilateralGaussianPermutohedral")
      .Input("image: T")
      .Input("stddev_spat: T")
      .Input("stddev_col: T")
      .Output("blurred: T")
      .Attr("T: {float32, double}") // implementation for single precision and double precision
      .Doc(R"doc(
  Performs a bilateral gaussian blur using a permutohedral lattice
  (see http://graphics.stanford.edu/papers/permutohedral/ for reference)
  )doc");


  REGISTER_OP("BilateralGaussianPermutohedralGrad")
        .Input("image: T")
        .Input("stddev_spat: T")
        .Input("stddev_col: T")
        .Output("blurred: T")
        .Attr("T: {float32, double}")  // implementation for single precision and double precision
      .Doc(R"doc(
  Calculates the gradient of a bilateral gaussian blur using a permutohedral lattice
  (see http://graphics.stanford.edu/papers/permutohedral/ for reference)
  )doc");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class BilateralGaussianPermutohedralOp : public OpKernel {
 public:
  explicit BilateralGaussianPermutohedralOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    printf("\n\n\n....Hello from BilateralGaussianPermutohedralOp Compute\n\n\n");

    // grab arguments
    const Tensor& image         = context->input(0);
    const Tensor stddev_spat    = context->input(1);
    const Tensor stddev_col     = context->input(2);

    // dimensionality checks
    OP_REQUIRES(context, image.shape().dims() == 3,
                errors::InvalidArgument("image must be 3-D, but got shape ",
                                        image.shape().DebugString()));

    // allocate output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &output_tensor));

    // convert input and output image to EIGEN::TensorMap<Eigen::Tensor<...>>
    eigen3tensorconst     input_eigen = image.tensor<float,3>();
    eigen3tensor          out_eigen   = output_tensor->tensor<float,3>();

    // convert parameters to floats
    const float spat_f = stddev_spat.scalar<float>()();
    const float col_f = stddev_col.scalar<float>()();

    // Filter the input with respect to the position vectors. (see permutohedral.h)
    PermutohedralLattice::filter(input_eigen, &out_eigen, spat_f , col_f , false);

  }
};


template <typename Device, typename T>
class BilateralGaussianPermutohedralGradOp : public OpKernel {
 public:
  explicit BilateralGaussianPermutohedralGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    printf("\n\n\n....Hello from BilateralGaussianPermutohedralGradOp Compute\n\n\n");

    // grab arguments
    const Tensor& image         = context->input(0);
    const Tensor stddev_spat    = context->input(1);
    const Tensor stddev_col     = context->input(2);

    // dimensionality checks
    OP_REQUIRES(context, image.shape().dims() == 3,
                errors::InvalidArgument("image must be 3-D, but got shape ",
                                        image.shape().DebugString()));


    // allocate output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &output_tensor));

    // convert input and output image to EIGEN::TensorMap<Eigen::Tensor<...>>
    eigen3tensorconst     input_eigen = image.tensor<float,3>();
    eigen3tensor          out_eigen = output_tensor->tensor<float,3>();


    // convert parameters to floats
    const float spat_f = stddev_spat.scalar<float>()();
    const float col_f = stddev_col.scalar<float>()();


    // Filter the input with respect to the position vectors. (see permutohedral.h)
    PermutohedralLattice::filter(input_eigen, &out_eigen, spat_f, col_f, true);


  }
};


// register kernels
#define REGISTER(Dev, T)                   \
REGISTER_KERNEL_BUILDER(                        \
    Name("BilateralGaussianPermutohedralGrad") \
        .Device(DEVICE_##Dev)                   \
        .TypeConstraint<T>("T"),                 \
    BilateralGaussianPermutohedralGradOp<Dev##Device, T>);


    REGISTER(CPU, float)
    REGISTER(CPU, double)
    REGISTER(CPU, Eigen::half)
#undef REGISTER


#define REGISTER(Dev, T)                   \
REGISTER_KERNEL_BUILDER(                        \
    Name("BilateralGaussianPermutohedral") \
        .Device(DEVICE_##Dev)                   \
        .TypeConstraint<T>("T"),                 \
    BilateralGaussianPermutohedralOp<Dev##Device, T>);


    REGISTER(CPU, float)
    REGISTER(CPU, double)
    REGISTER(CPU, Eigen::half)
#undef REGISTER






// REGISTER_KERNEL_BUILDER(Name("BilateralGaussianPermutohedral").Device(DEVICE_CPU), BilateralGaussianPermutohedralOp);
// REGISTER_KERNEL_BUILDER(Name("BilateralGaussianPermutohedralGrad").Device(DEVICE_CPU),BilateralGaussianPermutohedralGradOp);

}  // end namespace tensorflow
