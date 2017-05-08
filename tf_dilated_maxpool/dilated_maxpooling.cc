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
  }

  void Compute(OpKernelContext* context) override {

    printf(" Hello From DilatedMaxPoolingOp Compute\n");
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    if (params.depth_window > 1) { // this results in DepthwiseMaxPool (unneeded)
    } else {
      SpatialMaxPool(context, output, tensor_in, params, padding_);
    }
  }

 // private:


  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& tensor_in, const PoolParameters& params,
                      const Padding& padding) {
    // On GPU, use Eigen's Spatial Max Pooling.  On CPU, use an
    // EigenMatrix version that is currently faster than Eigen's
    // Spatial MaxPooling implementation.
    //
    // TODO(vrv): Remove this once we no longer need it.
    // if (std::is_same<Device, GPUDevice>::value) {
    //   Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);
    //   functor::SpatialMaxPooling<Device, T>()(
    //       context->eigen_device<Device>(), output->tensor<T, 4>(),
    //       tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
    //       params.row_stride, params.col_stride, pt);
    // } else {

    printf(" Hello From SpatialMaxPool\n");
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          ConstEigenMatrixMap;
      typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          EigenMatrixMap;

      ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                                 params.tensor_in_cols * params.tensor_in_rows *
                                     params.tensor_in_batch);
      EigenMatrixMap out_mat(
          output->flat<T>().data(), params.depth,
          params.out_width * params.out_height * params.tensor_in_batch);

      const DeviceBase::CpuWorkerThreads& worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());

      printf("1\n");

      // The following code basically does the following:
      // 1. Flattens the input and output tensors into two dimensional arrays.
      //    tensor_in_as_matrix:
      //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
      //    output_as_matrix:
      //      depth by (out_width * out_height * tensor_in_batch)
      //
      // 2. Walks through the set of columns in the flattened
      // tensor_in_as_matrix,
      //    and updates the corresponding column(s) in output_as_matrix with the
      //    max value.
      auto shard = [&params, &in_mat, &out_mat](int64 start, int64 limit) {

        const int32 in_rows = params.tensor_in_rows;
        const int32 in_cols = params.tensor_in_cols;
        const int32 pad_rows = params.pad_rows;
        const int32 pad_cols = params.pad_cols;
        const int32 window_rows = params.window_rows;
        const int32 window_cols = params.window_cols;
        const int32 row_stride = params.row_stride;
        const int32 col_stride = params.col_stride;
        const int32 out_height = params.out_height;
        const int32 out_width = params.out_width;

        {
          // Initializes the output tensor with MIN<T>.
          const int32 output_image_size = out_height * out_width * params.depth;
          EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
                                   1, (limit - start) * output_image_size);
          out_shard.setConstant(Eigen::NumTraits<T>::lowest());
        }
        printf("Hello from shard\n");

        for (int32 b = start; b < limit; ++b) {
          const int32 out_offset_batch = b * out_height;
          for (int32 h = 0; h < in_rows; ++h) {
            for (int32 w = 0; w < in_cols; ++w) {
              // (h_start, h_end) * (w_start, w_end) is the range that the input
              // vector projects to.
              const int32 hpad = h + pad_rows;
              const int32 wpad = w + pad_cols;
              const int32 h_start = (hpad < window_rows)
                                        ? 0
                                        : (hpad - window_rows) / row_stride + 1;
              const int32 h_end = std::min(hpad / row_stride + 1, out_height);
              const int32 w_start = (wpad < window_cols)
                                        ? 0
                                        : (wpad - window_cols) / col_stride + 1;
              const int32 w_end = std::min(wpad / col_stride + 1, out_width);
              // compute elementwise max
              const int32 in_offset = (b * in_rows + h) * in_cols + w;
              for (int32 ph = h_start; ph < h_end; ++ph) {
                const int32 out_offset_base =
                    (out_offset_batch + ph) * out_width;
                for (int32 pw = w_start; pw < w_end; ++pw) {
                  const int32 out_offset = out_offset_base + pw;
                  out_mat.col(out_offset) =
                      out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                }
              }
            }
          }
        }
      }; // auto shard
      printf("2\n");

      // TODO(andydavis) Consider sharding across batch x rows x cols.
      // TODO(andydavis) Consider a higher resolution shard cost model.
      const int64 shard_cost = params.tensor_in_rows * params.tensor_in_cols * params.depth;
      printf("shard_cost = %i\n", shard_cost);
      Shard(worker_threads.num_threads, worker_threads.workers,params.tensor_in_batch, shard_cost, shard); //segmentation fault

      printf("Bye from SpatialMaxPool\n");

    // } // if device is CPUDevice
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
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
