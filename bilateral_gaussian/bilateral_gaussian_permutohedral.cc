


#define EIGEN_USE_THREADS

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
      .Input("reverse: T")
      .Output("blurred: T")
      .Attr("T: {half, float, double}")
      .Doc(R"doc(
  Performs a bilateral gaussian blur using a permutohedral lattice
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

    // printf("Hello from BilateralGaussianPermutohedralOp Compute\n\n\n");

    // grab input variables
    const Tensor& image         = context->input(0);
    const Tensor stddev_spat    = context->input(1);
    const Tensor stddev_col     = context->input(2);
    const Tensor reverse        = context->input(3); // new

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

    // use inverse values for both standard deviations
    const float invSpatialStdev = 1.0f/spat_f;
    const float invColorStdev = 1.0f/col_f;

    // convert Tensor reverse to bool
    const int  reverse_int  = reverse.scalar<int>()(); // new
    const bool reverse_bool = (reverse_int == 1)? true:false;

    // Construct the position vectors out of x, y, r, g, and b.
    Tensor positions(DT_FLOAT, TensorShape({image.dim_size(0), image.dim_size(1), 5}));

    // again, convert Tensorflow::Tensor to EIGEN::TensorMap<Eigen::Tensor<...>>
    eigen3tensor positions_eigen = positions.tensor<float,3>();

    for (int y = 0; y < image.dim_size(1); y++) {
            for (int x = 0; x < image.dim_size(0); x++) {
                    positions_eigen(x, y, 0) = invSpatialStdev * x;
                    positions_eigen(x, y, 1) = invSpatialStdev * y;
                    positions_eigen(x, y, 2) = invColorStdev * input_eigen(x, y, 0);
                    positions_eigen(x, y, 3) = invColorStdev * input_eigen(x, y, 1);
                    positions_eigen(x, y, 4) = invColorStdev * input_eigen(x, y, 2);
            }
    }

    // Filter the input with respect to the position vectors. (see permutohedral.h)
    PermutohedralLattice::filter(input_eigen, positions_eigen, &out_eigen, reverse_bool);


  }
};

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



}  // end namespace tensorflow
