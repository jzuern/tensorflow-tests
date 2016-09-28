


#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
// #include "permutohedral.h"


namespace tensorflow {


  REGISTER_OP("BilateralGaussianPermutohedral")
      .Input("image: T_int")
      .Input("stddev_spat: T")
      .Input("stddev_col: T")
      .Output("blurred: T_int")
      .Attr("T: {half, float, double}")
      .Attr("T_int: {int32, int64} = DT_INT64")
      .Doc(R"doc(
  Performing a bilateral gaussian blur using an efficient approximation by using a permutohedral lattice structure (see paper for reference)

  )doc");






typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class BilateralGaussianPermutohedralOp : public OpKernel {
 public:
  explicit BilateralGaussianPermutohedralOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    printf("Hello from BilateralGaussianPermutohedralOp Compute\n");

    // grab input variables
    const Tensor& image = context->input(0);
    const Tensor stddev_spat = context->input(1);
    const Tensor stddev_col = context->input(2);

    // dimensionality checks
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(image.shape()),
                errors::InvalidArgument("image must be 3-D, but got shape ",
                                        image.shape().DebugString())); //jzuern: does not cover all cases


    // convert input image to flat EIGEN Tensor
    auto input = image.flat<int32>();

    // allocate output tensor
    Tensor* blurred_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &blurred_out));


    auto output = blurred_out->flat<int32>();
    auto spat = stddev_spat.flat<float>();
    float spat_f = spat(0); // TODO: this is terrible...

    auto col  = stddev_spat.flat<float>();
    float col_f = col(0);// TODO: this is terrible...

    const float invSpatialStdev = 1.0f/spat_f;
    const float invColorStdev = 1.0f/col_f;

    // Construct the position vectors out of x, y, r, g, and b.
    Tensor* positions(1, image.dim_size(0), image.dim_size(0), 5);


    // for (int y = 0; y < input.height; y++) {
    //         for (int x = 0; x < input.width; x++) {
    //                 positions(x, y)[0] = invSpatialStdev * x;
    //                 positions(x, y)[1] = invSpatialStdev * y;
    //                 positions(x, y)[2] = invColorStdev * input(x, y)[0];
    //                 positions(x, y)[3] = invColorStdev * input(x, y)[1];
    //                 positions(x, y)[4] = invColorStdev * input(x, y)[2];
    //         }
    // }

    // Filter the input with respect to the position vectors. (see permutohedral.h)
    // Image out = PermutohedralLattice::filter(input, positions);

    //
    // // We learn: we can access tensor elements of flattened-out tensor by simply
    // // calling them with operator()
    // const int N = input.size();
    // for (int i = 1; i < N; i++) {
    //   output(i) = i;
    //   printf("i = %i", i);
    // }


    // start Permutohedral code here...










  }
};





#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("BilateralGaussianPermutohedral") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("T_int"),      \
      BilateralGaussianPermutohedralOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64)


#undef REGISTER



}  // end namespace tensorflow
