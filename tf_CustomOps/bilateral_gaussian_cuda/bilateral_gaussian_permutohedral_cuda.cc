
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


void filter(const float *input, float *output, int pd, int vd, int w, int h, int nChannels, const float * spat_f, const float * col_f,
  float * ref,
  float * values_out,
  float * newValues,
  float * scaleFactor,
  float * table_test,
  float * pers
);// forward declaration of filter function


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


    auto input  = image.tensor<float,3>();
    auto output = output_tensor->tensor<float,3>();

      if (last_image_width == 0){ // this is true if Compute method is called the very first time

        printf("This is only executed for the very first frame\n");

        last_image_width = input.dimension(0);
        last_image_height = input.dimension(1);


        // allocate_persistent stuff that is needed for each kernel call (stays in memory after Kernel Compute Method is executed, and only the specific values are updated with each frame)
        shape = {10,10,10};
        OP_REQUIRES_OK(context,context->allocate_persistent(DT_FLOAT, shape, &persistent, nullptr));

        /* missing:
        ref
        values
        newValues
        scaleFactor
        ...

        */
  }

    // allocate filter tensor:
    TensorShape ref_shape({input.dimension(0), input.dimension(1), input.dimension(2) + 2});
    Tensor ref_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       ref_shape, &ref_tensor));
    auto ref = ref_tensor.tensor<float,3>();


    // allocate values tensor
    TensorShape values_shape({input.dimension(0), input.dimension(1), input.dimension(2)});
    Tensor values_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       values_shape, &values_tensor));
    auto values_out = values_tensor.tensor<float,3>();


    // allocate newValues tensor
    TensorShape newValues_shape({input.dimension(0), input.dimension(1), input.dimension(2) + 1, input.dimension(2) + 3});
    Tensor newValues_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       newValues_shape, &newValues_tensor));
    auto newValues = newValues_tensor.flat<float>();


    // allocate scaleFactor tensor
    TensorShape scaleFactor_shape({ input.dimension(2) + 1});
    Tensor scaleFactor_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       scaleFactor_shape, &scaleFactor_tensor));
    auto scaleFactor = scaleFactor_tensor.flat<float>();


    // allocate table_test tensor
    TensorShape table_test_shape({input.dimension(0), input.dimension(1), 2*(input.dimension(2) + 3)});
    Tensor table_test_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                       table_test_shape, &table_test_tensor));
    auto table_test = table_test_tensor.flat<float>();


  // Tensor spat_tensor, col_tensor;
  // OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, {1}, &spat_tensor));
  // OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, {1}, &col_tensor));


    // convert parameters to floats
    auto spat = stddev_spat.flat<float>();
    auto col  = stddev_col.flat<float>();

    const int height = input.dimension(0);
    const int width = input.dimension(1);
    const int nChannels = input.dimension(2);    // get number of color channels


    // get as persistent declared tensor from Kernel constructor
    Tensor* pers_tensor = persistent.AccessTensor(context);
    auto pers = pers_tensor->flat<float>();


    printf("Calling filter...\n");
    filter(input.data(), output.data(), 5, 3, width, height, nChannels,
    spat.data() ,
    col.data(),
    ref.data(),
    values_out.data(),
    newValues.data(),
    scaleFactor.data(),
    table_test.data(),
    pers.data());

  }


  PersistentTensor persistent;
  TensorShape shape;

  int last_image_width;
  int last_image_height;
};

REGISTER_KERNEL_BUILDER(Name("BilateralGaussianPermutohedralCuda").Device(DEVICE_GPU), BilateralGaussianPermutohedralCudaOp);
