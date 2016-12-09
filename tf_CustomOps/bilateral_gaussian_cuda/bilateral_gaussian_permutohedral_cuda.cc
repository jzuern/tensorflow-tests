
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

REGISTER_OP("BilateralGaussianPermutohedralCuda")
    .Input("image: T")
    .Input("stddev_spat: T")
    .Input("stddev_col: T")
    .Output("blurred: T")
    .Attr("T: {float32, double}") // implementation for single precision and double precision
    .Doc(R"doc(
Performs a bilateral gaussian blur using a permutohedral lattice
(see http://graphics.stanford.edu/papers/permutohedral/ for reference)
)doc");


void filter(const float *input, float *output, int pd, int vd, int w, int h, int nChannels, const float * spat_f, const float * col_f, bool accurate, float * test); // forward declaration of filter function

class BilateralGaussianPermutohedralCudaOp : public OpKernel {
 public:
  explicit BilateralGaussianPermutohedralCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    printf("\n\n\n....Hello from BilateralGaussianPermutohedralCudaOp Compute\n\n\n");

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
    auto input  = image.tensor<float,3>();
    auto output = output_tensor->tensor<float,3>();

    // allocate filter tensor:
    TensorShape test_shape({5000, 5000, 5});
    Tensor test_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       test_shape, &test_tensor));

    auto test = test_tensor.tensor<float,3>();
    int test_size = 5000*5000*5; // number of entries in tensor



    // convert parameters to floats
    auto spat = stddev_spat.flat<float>();
    auto col  = stddev_col.flat<float>();

    const int height = input.dimension(0);
    const int width = input.dimension(1);
    const int nChannels = input.dimension(2);    // get number of color channels

    const bool accurate = true; // make accurate blurring variance or not


    printf("Calling filter...\n");
    filter(input.data(), output.data(), 5, 3, width, height, nChannels, spat.data() ,col.data(), accurate, test.data());

  }
};

REGISTER_KERNEL_BUILDER(Name("BilateralGaussianPermutohedralCuda").Device(DEVICE_GPU), BilateralGaussianPermutohedralCudaOp);
