
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include <sys/time.h> // for gettimeofday



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
  float * values_table,
    int * entries_table,
  short * keys_table,
  float * matrix_float,
    int * matrix_int);


class BilateralGaussianPermutohedralCudaOp : public OpKernel {
 public:
  explicit BilateralGaussianPermutohedralCudaOp(OpKernelConstruction* context) : OpKernel(context) {

    // this is the constructor
    last_image_width = 0;
    last_image_height = 0;

  }

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

    int pd = input.dimension(2)+2;
    int vd = input.dimension(2);

    printf("pd = %i, vd = %i\n", pd,vd);

      if ((last_image_width == 0) ||  (last_image_width != input.dimension(0)) ){ // this is true if Compute method is called the very first time or image resolution changes

        printf("This is only executed for the very first frame or when image dimensions change\n");

        last_image_width = input.dimension(0);
        last_image_height = input.dimension(1);

        // allocate_persistent stuff that is needed for each kernel call (stays in memory after Kernel Compute Method is executed, and only the specific values are updated with each frame)

        timeval t2[2];
        gettimeofday(t2+0, NULL);

        // allocate ref tensor:
        printf("allocating persistent ref tensor \n");
        ref_shape = {input.dimension(0), input.dimension(1), pd};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,ref_shape, &ref_tensor, nullptr));

        // allocate values tensor:
        printf("allocating persistent values tensor \n");
        values_shape = {input.dimension(0), input.dimension(1), vd};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,values_shape, &values_tensor, nullptr));

        // allocate newValues tensor:
        printf("allocating persistent newValues tensor \n");
        newValues_shape = {input.dimension(0), input.dimension(1), vd+1 , pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,newValues_shape, &newValues_tensor, nullptr));


        // allocate scaleFactor tensor:
        printf("allocating persistent scaleFactor tensor \n");
        scaleFactor_shape = { vd+1 };
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,scaleFactor_shape, &scaleFactor_tensor, nullptr));


        // allocate values_table tensor:
        printf("allocating persistent values_table tensor \n");
        values_table_shape = {input.dimension(0), input.dimension(1), pd,pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,values_table_shape, &values_table_tensor, nullptr));

        // allocate entries_table tensor:
        printf("allocating persistent entries_table tensor \n");
        entries_table_shape = {input.dimension(0), input.dimension(1), 2 ,pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT32,entries_table_shape, &entries_table_tensor, nullptr));

        // allocate keys_table tensor:
        printf("allocating persistent keys_table tensor \n");
        keys_table_shape = {input.dimension(0), input.dimension(1), pd,pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT16,keys_table_shape, &keys_table_tensor, nullptr));

        // allocate matrix_int tensor:
        printf("allocating persistent keys_table tensor \n");
        matrix_int_shape = {input.dimension(0), input.dimension(1), pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT32,matrix_int_shape, &matrix_int_tensor, nullptr));

        // allocate matrix_float tensor:
        printf("allocating persistent keys_table tensor \n");
        matrix_float_shape = {input.dimension(0), input.dimension(1), pd+1};
        OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT,matrix_float_shape, &matrix_float_tensor, nullptr));

        gettimeofday(t2+1, NULL);
    	  printf("allocating with allocate_persistent took %3.3f ms\n", (t2[1].tv_sec - t2[1-1].tv_sec)*1000.0 + (t2[1].tv_usec - t2[1-1].tv_usec)/1000.0);




  } else{ // current image dims same as last

    // do nothing

  }



    // get Tensors from allocate_persistent
    timeval t1[2];
    gettimeofday(t1+0, NULL);

    Tensor* ref_persistent = ref_tensor.AccessTensor(context);
    auto ref = ref_persistent->flat<float>();

    Tensor* values_persistent = values_tensor.AccessTensor(context);
    auto values = values_persistent->flat<float>();

    Tensor* newValues_persistent = newValues_tensor.AccessTensor(context);
    auto newValues = newValues_persistent->flat<float>();

    Tensor* scaleFactor_persistent = scaleFactor_tensor.AccessTensor(context);
    auto scaleFactor = scaleFactor_persistent->flat<float>();

    Tensor* values_table_persistent = values_table_tensor.AccessTensor(context);
    auto values_table = values_table_persistent->flat<float>();

    Tensor* entries_table_persistent = entries_table_tensor.AccessTensor(context);
    auto entries_table = entries_table_persistent->flat<int>();

    Tensor* keys_table_persistent = keys_table_tensor.AccessTensor(context);
    auto keys_table = keys_table_persistent->flat<short>();

    Tensor* matrix_int_persistent = matrix_int_tensor.AccessTensor(context);
    auto matrix_int = matrix_int_persistent->flat<int>();

    Tensor* matrix_float_persistent = matrix_float_tensor.AccessTensor(context);
    auto matrix_float = matrix_float_persistent->flat<float>();

    // gettimeofday(t1+1, NULL);
	  // printf("%3.3f ms\n", (t1[1].tv_sec - t1[1-1].tv_sec)*1000.0 + (t1[1].tv_usec - t1[1-1].tv_usec)/1000.0);


    // convert parameters to floats
    auto spat = stddev_spat.flat<float>();
    auto col  = stddev_col.flat<float>();

    const int height = input.dimension(0);
    const int width = input.dimension(1);
    const int nChannels = input.dimension(2);    // get number of color channels



    printf("Calling filter...\n");
    filter(input.data(), output.data(), pd, vd, width, height, nChannels,
    spat.data() ,
    col.data(),
    ref.data(),
    values.data(),
    newValues.data(),
    scaleFactor.data(),
    values_table.data(),
    entries_table.data(),
    keys_table.data(),
    matrix_float.data(),
    matrix_int.data());
  }


  PersistentTensor ref_tensor;
  PersistentTensor values_tensor;
  PersistentTensor newValues_tensor;
  PersistentTensor scaleFactor_tensor;
  PersistentTensor values_table_tensor;
  PersistentTensor entries_table_tensor;
  PersistentTensor keys_table_tensor;
  PersistentTensor matrix_int_tensor;
  PersistentTensor matrix_float_tensor;


  TensorShape ref_shape;
  TensorShape values_shape;
  TensorShape newValues_shape;
  TensorShape scaleFactor_shape;
  TensorShape values_table_shape;
  TensorShape entries_table_shape;
  TensorShape keys_table_shape;
  TensorShape matrix_int_shape;
  TensorShape matrix_float_shape;


  int last_image_width;
  int last_image_height;
};

REGISTER_KERNEL_BUILDER(Name("BilateralGaussianPermutohedralCuda").Device(DEVICE_GPU), BilateralGaussianPermutohedralCudaOp);
