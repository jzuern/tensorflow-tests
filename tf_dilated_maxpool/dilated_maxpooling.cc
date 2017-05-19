#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/maxpooling_op.h"

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"



typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;


REGISTER_OP("DilatedMaxPool")
    .Attr("T: realnumbertype = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("dilation_rate: int >= 1")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape)
    .Doc(R"doc(
Performs max pooling on the input.
ksize: The size of the window for each dimension of the input tensor.
strides: The stride of the sliding window for each dimension of the
  input tensor.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
input: 4-D input to pool over.
output: The max pooled output tensor.
)doc");



template <typename Device, typename T>
class DilatedMaxPoolingOp : public OpKernel {
 public:
  explicit DilatedMaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default DilatedMaxPoolingOp only supports NHWC."));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_rate", &dilation_rate_));

  }

  void Compute(OpKernelContext* context) override {

    printf(" Hello From DilatedMaxPoolingOp Compute\n");
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }



    // Create an output tensor
    Tensor* output_tensor = NULL;

    TensorShape out_shape({params.tensor_in_batch, params.out_height, params.out_width, params.depth});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,&output_tensor));

    SpatialMaxPool(context, output_tensor, tensor_in, params, dilation_rate_);

  } // void Compute



  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& tensor_in, const PoolParameters& params, const int dilation_rate) {


          // if padding == "VALID":no padding
          // if padding == "SAME": Input and output layers have the same size.

          // Tests:
          // - non-quadratic image input .. success
          // - 3 color channels .. success
          // - test stride != window_size .. success
          //- dilation_rate > 1 .. success
          // Bugs:
          // - if multiple color channels: values get mixed up... fixed


        auto test_in = tensor_in.tensor<float,4>(); // force conversion to float
        auto test_out = output->tensor<float,4>();// force conversion to float

        const int nBatch = tensor_in.dim_size(0);
        printf("nBatch = %i\n", nBatch);

        const int width = tensor_in.dim_size(1);
        printf("width = %i\n", width);

        const int height = tensor_in.dim_size(2);
        printf("height = %i\n", height);

        const int nChannels = tensor_in.dim_size(3);
        printf("nChannels = %i\n", nChannels);

        const int N = test_in.size();
        printf("N = %i\n", N);

        int dilationH = dilation_rate; // dilation_rate height
        int dilationW = dilation_rate; // dilation_rate width
        int dH = params.row_stride; // stride in height
        int dW = params.col_stride; // stride in width
        int in_height = params.tensor_in_rows; // input tensor height
        int in_width = params.tensor_in_cols; // input tensor width
        int out_height = params.out_height; // input tensor height
        int out_width = params.out_width; // input tensor width
        int padH = 0;                        // padding in height --> is zero by default
        int padW = 0;                         // padding in width --> is zero by default
        int kH = params.window_rows; // window size in height
        int kW = params.window_cols; // window size in width


        printf("PoolParameters: \n\nin_height = %i, in_width = %i,  \n"
                                          "window_rows = %i, window_cols = %i \n"
                                          "row_stride = %i, col_stride = %i, \n"
                                          "out_height = %i, out_width = %i, \n",
                                          in_height, in_width, kH ,kW , dH,  dW, out_height,out_width );



        for (int b = 0; b < nBatch; b++) { // batch
          for (int c = 0; c < nChannels; c++) { // channel
            for (int i = 0; i < params.out_height; i++) { // heigth of output image
              for (int j = 0; j < params.out_width; j++) { // width of output image

                int hstart = i * dH - padH;
                int wstart = j * dW - padW;
                int hend = fminf(hstart + (kH - 1) * dilationH + 1, in_height);
                int wend = fminf(wstart + (kW - 1) * dilationW + 1, in_width);
                while(hstart < 0) hstart += dilationH;
                while(wstart < 0) wstart += dilationW;

                float maxval = -INFINITY;

                int x,y;
                for(y = hstart; y < hend; y += dilationH){
                  for(x = wstart; x < wend; x += dilationW){
                    float val = test_in(b,y,x,c);
                    if (val > maxval) maxval = val;
                  }
                }

                /* set output to local max */
                test_out(b,i,j,c) = maxval;
              }
            }
          }
        }

        // // print input
        // printf("input image: \n");
        //
        // for(int i = 0; i < params.tensor_in_rows; i++){
        //   for(int j = 0; j < params.tensor_in_cols; j++){
        //     printf("%f ", test_in(0,i,j,0));
        //   }
        //   printf("\n");
        // }
        // printf("output image: \n");
        //
        // for(int i = 0; i < params.out_height; i++){
        //   for(int j = 0; j < params.out_width; j++){
        //     printf("%f ", test_out(0,i,j,0));
        //   }
        //   printf("\n");
        // }

  } // void SpatialMaxPool

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  int dilation_rate_;
};


#define REGISTER_MAX_POOL_KERNELS(D, T)                                  \

// Below kernels implemented only for CPU device.
#define REGISTER_CPU_ONLY_POOL_KERNELS(T)                        \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DilatedMaxPool").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DilatedMaxPoolingOp<CPUDevice, T>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_ONLY_POOL_KERNELS);
#undef REGISTER_CPU_ONLY_POOL_KERNELS

#define REGISTER_CPU_MAX_POOL_KERNELS(T) REGISTER_MAX_POOL_KERNELS(CPU, T);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_MAX_POOL_KERNELS);
#undef REGISTER_CPU_KERNELS


#undef REGISTER_MAX_POOL_KERNELS
