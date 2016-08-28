

#include ....



// UnsortedSegmentSumFunctor implementation for CPUDevice.
template <typename T, typename Index>
struct UnsortedSegmentSumFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) {
    output.setZero();
    if (data_size == 0) {
      return;
    }
    const int64 N = segment_ids.dimension(0);
    auto data_flat = typename TTypes<T, 2>::ConstTensor(data, N, data_size / N);
    for (int64 i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      OP_REQUIRES(ctx, FastBoundsCheck(j, output_rows),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", output_rows, ")"));
      output.template chip<0>(j) += data_flat.template chip<0>(i);
    }
  }
};




// Similar to SegmentReductionOp but can handle unsorted segment definitions and
// specifying size of output.
template <typename Device, class T, class Index>
class UnsortedSegmentSumOp : public OpKernel {
 public:
  explicit UnsortedSegmentSumOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    const Tensor& data = context->input(0); // python: data
    const Tensor& segment_ids = context->input(1); // python: segment_ids
    const Tensor& num_segments = context->input(2); // python: num_segments

    OP_REQUIRES(
        context, IsLegacyScalar(num_segments.shape()),
        errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                num_segments.shape().DebugString()));
    OP_REQUIRES(
        context,
        TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
        errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                                " does not start with segment_ids.shape = ",
                                segment_ids.shape().DebugString()));

    const auto segment_flat = segment_ids.flat<Index>();
    const Index output_rows =
        internal::SubtleMustCopy(num_segments.scalar<int32>()());
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));


    TensorShape output_shape; // initialize output_shape
    output_shape.AddDim(output_rows); // 0th dim of output Tensor has size output_rows ( == num_segments)
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }

    Tensor* output = nullptr; // initialize output Tensor to nullpointer
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_flat = output->flat_outer_dims<T>(); // what means flat_outer_dims?

    auto data_ptr = data.template flat<T>().data();
    // now perform unsortedSegmentSum with functor:

    
    functor::UnsortedSegmentSumFunctor<Device, T, Index>()(
        context, context->template eigen_device<Device>(), output_rows,
        segment_ids.shape(), segment_flat, data.NumElements(), data_ptr,
        output_flat);
  }
};

};
