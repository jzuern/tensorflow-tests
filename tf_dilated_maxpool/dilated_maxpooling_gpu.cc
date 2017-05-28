/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


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

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("DilatedMaxPooling")
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

void DilatedMaxPoolingKernelLauncher(const float* test_in, const int N, float* test_out, const int dilationH, const int dilationW,
                                const int dH, const int dW, const int in_height, const int in_width, const int nBatch, const int nChannels,
                                const int out_height, const int out_width, const int padH, const int padW, const int kH, const int kW);

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



    printf(" Hello From DilatedMaxPoolingOp CUDA Compute\n");
    const Tensor& in = context->input(0);
    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, in.shape()};
    if (!context->status().ok()) {
      return;
    }

    auto tensor_in = in.tensor<float,4>(); // force conversion to float

    // Create an output tensor
    Tensor* output_tensor = NULL;

    TensorShape out_shape({params.tensor_in_batch, params.out_height, params.out_width, params.depth});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,&output_tensor));

    auto tensor_out = output_tensor->tensor<float,4>();// force conversion to float

    const int nBatch = in.dim_size(0);
    printf("nBatch = %i\n", nBatch);
    const int width = in.dim_size(1);
    printf("width = %i\n", width);
    const int height = in.dim_size(2);
    printf("height = %i\n", height);
    const int nChannels = in.dim_size(3);
    printf("nChannels = %i\n", nChannels);
    const int N = tensor_in.size();
    printf("N = %i\n", N);

    int dilationH = dilation_rate_; // dilation_rate height
    int dilationW = dilation_rate_; // dilation_rate width
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


    // Call the cuda kernel launcher
    DilatedMaxPoolingKernelLauncher(tensor_in.data(), N, tensor_out.data(), dilationH, dilationW,dH,dW,in_height,in_width,nBatch,nChannels,out_height,out_width,padH,padW,kH,kW);

  } // void Compute


  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  int dilation_rate_;
};

REGISTER_KERNEL_BUILDER(Name("DilatedMaxPooling").Device(DEVICE_GPU), DilatedMaxPoolingOp);
