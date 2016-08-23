

#include ....



// Static routines not in the templated class to reduce code size
static void SegmentReductionValidationHelper(OpKernelContext* context,
                                             const Tensor& input,
                                             const Tensor& segment_ids) {
  OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
              errors::InvalidArgument("segment_ids should be a vector."));
  const int64 num_indices = segment_ids.NumElements();
  OP_REQUIRES(context, num_indices == input.dim_size(0),
              errors::InvalidArgument(
                  "segment_ids should be the same size as dimension 0 of"
                  " input."));
}





// This operator handles reducing segments along the first dimension.
// See core/ops/math_ops.cc for more details.
template <typename Device, class T, class Index, typename Reducer>
class SegmentReductionOp : public OpKernel {

  explicit SegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context) {}


public:

  void Compute(OpKernelContext* context) override {...}


  // ...
};



#define REGISTER_CPU_KERNELS // stuff


// Similar to SegmentReductionOp but can handle unsorted segment definitions and
// specifying size of output.
template <typename Device, class T, class Index>
class UnsortedSegmentSumOp : public OpKernel {


public:
 explicit UnsortedSegmentSumOp(OpKernelConstruction* context)
     : OpKernel(context) {}

 void Compute(OpKernelContext* context) override {

 const Tensor& data = context->input(0); // all data in Tensor
 const Tensor& segment_ids = context->input(1); // segment ids
 const Tensor& num_segments = context->input(2); // number of different segments

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
 const int64 N = segment_flat.dimension(0);
 const Index output_rows =
     internal::SubtleMustCopy(num_segments.scalar<int32>()());
 OP_REQUIRES(context, output_rows >= 0,
             errors::InvalidArgument("Input num_segments == ", output_rows,
                                     " must not be negative."));

 // write TensorShape shape of output Tensor
 TensorShape output_shape;
 output_shape.AddDim(output_rows); // 0th dimension of size num_segments
 for (int i = segment_ids.dims(); i < data.dims(); i++) {
   output_shape.AddDim(data.dim_size(i));
 }

 // initialize outputTensor to nullpointer
 Tensor* output = nullptr;
 OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
 auto output_flat = output->flat_outer_dims<T>();
 output_flat.setZero();

 if (data.NumElements() > 0) {
   auto data_flat = data.shaped<T, 2>({N, data.NumElements() / N});
   for (int64 i = 0; i < N; ++i) {
     Index j = internal::SubtleMustCopy(segment_flat(i));
     OP_REQUIRES(context, FastBoundsCheck(j, output_rows),
                 errors::InvalidArgument(
                     "segment_ids", SliceDebugString(segment_ids.shape(), i),
                     " = ", j, " is out of range [0, ", output_rows, ")"));
     output_flat.template chip<0>(j) += data_flat.template chip<0>(i); // note the ".template" suffix on chip<0>
   }
 }
}

};


// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionOpBase : public OpKernel {

public:
 explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                       bool is_mean, bool is_sqrtn)
     : OpKernel(context), is_mean_(is_mean), is_sqrtn_(is_sqrtn) {}

 void Compute(OpKernelContext* context) override {...}


private:

  int64 Reduce(const typename TTypes<T>::ConstMatrix& input_flat,
               const typename TTypes<Index>::ConstVec& indices_vec, int64 start,
               int64 num,
               Eigen::TensorChippingOp<0, typename TTypes<T>::Matrix> out) {...}



};


template <typename Device, class T>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, true /*is_mean*/,
                                                false /*is_sqrtn*/) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, false /*is_mean*/,
                                                true /*is_sqrtn*/) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, false /*is_mean*/,
                                                false /*is_sqrtn*/) {}
};


template <class T>
class SparseSegmentGradOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradOpBase(OpKernelConstruction* context, bool is_sqrtn)
      : OpKernel(context), is_sqrtn_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override { ...}

 private:


};



template <class T>
class SparseSegmentMeanGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, false /*is_sqrtn*/) {}
};

template <class T>
class SparseSegmentSqrtNGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, true /*is_sqrtn*/) {}
};
