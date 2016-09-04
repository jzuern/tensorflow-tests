    /* important source files and classes/structs:


    TensorBase.h:
      - TensorBase.sum method implementation
      - Describes behavior of Tensor base class (on fundamendal level)


    TensorReduction.h:
      - Method: firstInput(...)
      - struct: DimInitializer
        - Method: run(...)
      - struct GenericDimReducer(1 and 2)
      - struct TensorEvaluator
        - Method: packet(...)
        - Method: coeff(...)


    TensorFunctors.h
      - struct: SumReducer
        - method: reduce(...)

    TensorExecutor.h
      - struct: TensorExecutor
        - method: run(...)

    TensorEvaluator.h
      - struct: TensorEvaluator
        - method: coeff(...)

    
    
    */

    ////////////////////////////////////////////////////////////////////////
    // Information Flow when reducing a Tensor along some of its dimensions
    ////////////////////////////////////////////////////////////////////////

    // call TensorBase.sum(...)
    //    sum() returns Object of type TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>

    
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>
    sum(const Dims& dims) const {
      std::cout << "class TensorBase: TensorReductionOp sum(...)\n";  // jzuern
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::SumReducer<CoeffReturnType>());
    }


    // This is the constructor of struct TensorReductionOp
    // With its initializer list m_expr (??), m_dims (dimensions of tensor) , m_reducer (Sum reducer of type float)

    TensorReductionOp(const XprType& expr, const Dims& dims, const Op& reducer) : m_expr(expr), m_dims(dims), m_reducer(reducer)
    { 

      // .....

    }



    // we then call the constructor of class TensorExecutor and run its method void run(...):

    // IS THIS ONLY ALLOCATING THE TENSOR OBJECT IN MEMORY!!! (not important for reduction)


template<typename Expression>
class TensorExecutor<Expression, DefaultDevice, true>
{
 public:

  static inline void run(const Expression& expr, const DefaultDevice& device = DefaultDevice())
  {


    TensorEvaluator<Expression, DefaultDevice> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      const int PacketSize = unpacket_traits<typename TensorEvaluator<Expression, DefaultDevice>::PacketReturnType>::size;
      // Give the compiler a strong hint to unroll the loop. But don't insist
      // on unrolling, because if the function is expensive the compiler should not
      // unroll the loop at the expense of inlining.
      const Index UnrolledSize = (size / (4 * PacketSize)) * 4 * PacketSize;

      for (Index i = 0; i < UnrolledSize; i += 4*PacketSize) {
        for (Index j = 0; j < 4; j++) {
          evaluator.evalPacket(i + j * PacketSize);
        }
      }

      const Index VectorizedSize = (size / PacketSize) * PacketSize;
      for (Index i = UnrolledSize; i < VectorizedSize; i += PacketSize) {
        evaluator.evalPacket(i);
      }
      
      for (Index i = VectorizedSize; i < size; ++i) {
        evaluator.evalScalar(i);

      }
      std::cout << std::endl;
    }
    evaluator.cleanup();
  }
};

// As we can see, here we instantiate an object of type TensorEvaluator
// ??: What do we need the 3 loops for? why not 1 sufficient?


     TensorEvaluator(const XprType& op, const Device& device) : m_impl(op.expression(), device), m_reducer(op.reducer()), m_result(NULL), m_device(device)
  {

    // Build the bitmap indicating if an input dimension is reduced or not.
    
    // run DimInitializer in order to initialize dimensions of Output-Tensor 

    // Precompute output strides. (how many values does output Tensor have? )

    // Precompute input strides. (how many values of input Tensor do we have to access?)


  }


    // we then call EvalSubExprsIfNeeded(...) 
    // EvalSubExprsIfNeeded decides whether we use

    // we then  call coeff(...) :
    // coeff is being called as many times as there are entries in the output tensor.
    // coeff returns the value that has output tensor entry with Index index


    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {


      // initializing reducer: this initializes the accum variable which after reduction operations completed
      // holds the value of a specific entry of the output tensor 

      typename Self::CoeffReturnType accum = reducer.initialize(); 

      // call reduce function of GenericDimReducer on 

      // calling firstInput(index)
      internal::GenericDimReducer<NumReducedDims-1, Self, Op>::reduce(*this, firstInput(index), reducer, &accum);


      // firstInput: Returns the Index in the input tensor of the first value that needs to be
          // used to compute the reduction at output index "index".


      // finalizing accum (which is the {float, half, ...} value of the output tensor entry we wanted to calculate)
      return reducer.finalize(accum);
    }


    // GenericDimReducer:
    // GenericDimReducer runs the method reduce(...) of another instantiation of GenericDimReducer, which does get
    // an integer DimIndex as template parameter, but which is decreased by a value of 1. This happens until DimIndex == 0...



template <int DimIndex, typename Self, typename Op>
struct GenericDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    std::cout << "in struct GenericDimReducer1: *in* reduce() "//
      "with DimIndex == " << DimIndex <<  std::endl; // jzuern
    EIGEN_STATIC_ASSERT((DimIndex > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      GenericDimReducer<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
    std::cout << "in struct GenericDimReducer1: *out* reduce()" << std::endl; // jzuern

  }
};


// ... in which case this GenericDimReducer is being instantiated. This struct calls reducer.reduce, which in our case is SumReducer. 

template <typename Self, typename Op>
struct GenericDimReducer<0, Self, Op> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    std::cout << "in struct GenericDimReducer2: *in* reduce()" << std::endl; // jzuern
    for (int j = 0; j < self.m_reducedDims[0]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[0];
      reducer.reduce(self.m_impl.coeff(input), accum);
    }
    std::cout << "in struct GenericDimReducer2: *out* reduce()" << std::endl; // jzuern

  }
};



// This is SumReducer:

template <typename T> struct SumReducer
{
  static const bool PacketAccess = packet_traits<T>::HasAdd;
  static const bool IsStateful = false;

  // This is where the addition actually takes place
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    (*accum) += t; //summation
    std::cout << "in struct SumReducer reduce(...). accum = " << (*accum) << "\n"; // jzuern. here we perform the addition
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = padd<Packet>(*accum, p);
    std::cout << "in struct SumReducer reducePacket(...)\n"; // jzuern. here we perform the addition

  }

  // .....

};