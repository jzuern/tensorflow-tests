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
      printf( "class TensorBase: TensorReductionOp sum(...) (1)\n");  // jzuern
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::SumReducer<CoeffReturnType>());
    }




// As we can see, here we instantiate an object of type TensorEvaluator
// ??: What do we need the 3 loops for? why not 1 sufficient?


     TensorEvaluator(const XprType& op, const Device& device) : m_impl(op.expression(), device), m_reducer(op.reducer()), m_result(NULL), m_device(device)
  {

    // Build the bitmap indicating if an input dimension is reduced or not.
    
    // run DimInitializer in order to initialize dimensions of Output-Tensor 

    // Precompute output strides. (how many values does output Tensor have? )

    // Precompute input strides. (how many values of input Tensor do we have to access?)


  }


    // we then call EvalSubExprsIfNeeded(...) where we find out whether optimized version of Reducer is 
    // implemented and what dimensions are going to be reduced (inner, innermost, outer, outermost,...)
    // This affects whether a CPU Reducer or GPU Reducer is going to be launched

    // here we decide that we launch the full reduction: 


template <typename Self, typename Op, bool PacketAccess>
struct FullReductionLauncher<Self, Op, float, PacketAccess> {
  static void run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs) {

          printf("in CUDA FullReductionLauncher::run(...) \n");

    typedef typename Self::Index Index;
    typedef typename Self::CoeffReturnType Scalar;
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = divup<int>(num_coeffs, block_size * num_per_thread);

    unsigned int* semaphore = NULL;
    if (num_blocks > 1) {
      semaphore = device.semaphore();
    }

    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs, output, semaphore);

    printf("out CUDA FullReductionLauncher::run(...) \n");

  }
};

    // or i.e. an outer Reduction which invokes the OuterReducer: 

    struct OuterReducer<Self, Op, GpuDevice> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename Device, typename OutputType>
  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const Device&, OutputType*, typename Self::Index, typename Self::Index) {
    assert(false && "Should only be called to reduce floats on a gpu device");
    return true;
  }

  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;


  printf("*in* CUDA OuterReducer::run\n");

    // It's faster to use the usual code.
    if (num_coeffs_to_reduce <= 32) {
        printf("less than 32 coeffs to reduce, therefore no CUDA launch.\n");

      return true;
    }

    const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = 256;
    const int num_per_thread = 16;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs in the reduction kernel itself when we don't have to worry
      // about race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumCudaMultiProcessors() *
                             device.maxCudaThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
                         num_blocks, 1024, 0, device, reducer.initialize(),
                         num_preserved_vals, output);
    }

    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs_to_reduce, num_preserved_vals, output);

  printf("*out* CUDA OuterReducer::run\n");

    return false;
  }
};




  // this launches a full reduction kernel.....:






template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(Reducer reducer, const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output, unsigned int* semaphore) {
  // Initialize the output value
  const Index first_index = blockIdx.x * BlockSize * NumPerThread + threadIdx.x;
  if (gridDim.x == 1) {
    if (first_index == 0) {
      *output = reducer.initialize();
    }
  }
  else {
    if (threadIdx.x == 0) {
      unsigned int block = atomicCAS(semaphore, 0u, 1u);
      if (block == 0) {
        // We're the first block to run, initialize the output value
        atomicExch(output, reducer.initialize());
        __threadfence();
        atomicExch(semaphore, 2u);
      }
      else {
        // Wait for the first block to initialize the output value.
        // Use atomicCAS here to ensure that the reads aren't cached
        unsigned int val;
        do {
          val = atomicCAS(semaphore, 2u, 2u);
        }
        while (val < 2u);
      }
    }
  }

  __syncthreads();

  eigen_assert(gridDim.x == 1 || *semaphore >= 2u);

  typename Self::CoeffReturnType accum = reducer.initialize();
  Index max_iter = numext::mini<Index>(num_coeffs - first_index, NumPerThread*BlockSize);
  for (Index i = 0; i < max_iter; i+=BlockSize) {
    const Index index = first_index + i;
    eigen_assert(index < num_coeffs);
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

#pragma unroll

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reduce(__shfl_down(accum, offset, warpSize), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }

  if (gridDim.x > 1 && threadIdx.x == 0) {
    // Let the last block reset the semaphore
    atomicInc(semaphore, gridDim.x + 1);
  }

}




    // .... or an OuterReductionKernel




    template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void OuterReductionKernel(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                     typename Self::CoeffReturnType* output) {

  const Index num_threads = blockDim.x * gridDim.x;
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (gridDim.x == 1) {
    for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
      output[i] = reducer.initialize();
    }
    __syncthreads();
  }

  // Do the reduction.
  const Index max_iter = num_preserved_coeffs * divup<Index>(num_coeffs_to_reduce, NumPerThread);
  for (Index i = thread_id; i < max_iter; i += num_threads) {
    const Index input_col = i % num_preserved_coeffs;
    const Index input_row = (i / num_preserved_coeffs) * NumPerThread;
    typename Self::CoeffReturnType reduced_val = reducer.initialize();
    const Index max_row = numext::mini(input_row + NumPerThread, num_coeffs_to_reduce);
    for (Index j = input_row; j < max_row; j++) {
      typename Self::CoeffReturnType val = input.m_impl.coeff(j * num_preserved_coeffs + input_col);
      reducer.reduce(val, &reduced_val);

    }
    atomicReduce(&(output[input_col]), reduced_val, reducer);
  }


}


  // !: if we use more than 1 block on device, then we use a ReductionInitKernel in order to parallelize initialization of the 
  //    reduction stuff as well




  // each Kernel then calls the specific reduction function of the specific reducer (reducer.reduce(....))...


  /* Questions:

    - why does not everything in kernel print out?   (printf(....))
    - how do we reduce after kernel have run?






  */

