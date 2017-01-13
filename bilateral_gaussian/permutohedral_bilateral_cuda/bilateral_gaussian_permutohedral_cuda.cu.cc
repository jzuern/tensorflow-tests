

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#define BLOCK_SIZE 64
#define _DEBUG


#include "cutil.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif



// hash_table.cu
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////


#define USE_CUSTOM_MODULO

__device__ __constant__ float *table_values;
__device__ __constant__ signed short *table_keys;
__device__ __constant__ int *table_entries;
__device__ __constant__ unsigned int table_capacity;
__device__ __constant__ signed short *table_zeros;
__device__ __constant__ char *table_rank;


/*************************************************************/
/* Fast computation of modulo operator with constant divisor */
/*************************************************************/
__device__ __constant__ unsigned int __div_m;
__device__ __constant__ unsigned int __div_l;
__device__ __constant__ unsigned int __div_c;


#ifdef USE_CUSTOM_MODULO
__device__ inline unsigned int modHash(unsigned int n) {
  unsigned int t1 = __umulhi(__div_m, n);
  return n - ((t1+((n-t1)>>1))>>(__div_l-1)) * __div_c;
}

#else
#define modHash(n) ((n)%(2*table_capacity));
#endif

/*************************************************************/
/* End modulo                                                */
/*************************************************************/

__device__ __constant__ unsigned int hOffset[64];

// allocate hash table pointers on global scope


template<int pd, int vd>
void createHashTable(int capacity, float * values_table, int * entries_table, short * keys_table) {

    // capacity =  nPixel*(pd+1) = nPixel*6
    // vd = 3
    // pd = 5

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_capacity,&capacity,sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,  &values_table,sizeof(float *)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_entries,  &entries_table,sizeof(unsigned int *)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_keys,  &keys_table, sizeof(unsigned int *)));

}

template <int vd> static void resetHashTable() {
  CUDA_SAFE_CALL(cudaMemset((void*)table_values, 0, table_capacity*vd*sizeof(float)));
}


template<int kd> __device__ __host__ static unsigned int hash(signed short *key) {
    unsigned int k = 0;
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011;
    }
    return k;
}

template<int kd> __device__ __host__ static unsigned int hash(int *key) {
    unsigned int k = 0;
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011;
    }
    return k;
}

template<int d> __device__ static bool matchKey(int idx, signed short * key) {
    bool match = true;
    int slot = idx/(d+1), color = idx-slot*(d+1);
    char *rank = table_rank + slot * (d+1);
    signed short *zero = table_zeros + slot * (d+1);
    for (int i = 0; i < d && match; i++) {
	match = (key[i] == zero[i] + color - (rank[i] > d-color ? (d+1) : 0));
    }
    return match;
}


template<int d> __device__ static void generateKey(int idx, signed short * key) {
    int slot = idx/(d+1), color = idx-slot*(d+1);
    char *rank = table_rank + slot * (d+1);
    signed short *zero = table_zeros + slot * (d+1);
    for (int i = 0; i < d; i++) {
	key[i] = zero[i] + color - (rank[i] > d-color ? (d+1) : 0);
    }
}

static float* swapHashTableValues(float *newValues) {
    float * oldValues;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&oldValues,
					table_values,
					sizeof(float *)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
				      &newValues,
				      sizeof(float *)));
    return oldValues;
}


template<int kd>
__device__ static int hashTableInsert(unsigned int fh, signed short *key, unsigned int slot) {
    int h = modHash(fh);
    while (1) {
	int *e = &table_entries[h];

	// If the cell is empty (-1), lock it (-2)
	int contents = atomicCAS(e, -1, -2);

	if (contents == -2) {
	    // If it was locked already, move on to the next cell

	} else if (contents == -1) {
	    // If it was empty, we successfully locked it. Write our key.

	    for (int i = 0; i < kd; i++) {
		table_keys[slot*kd+i] = key[i];
	    }

	    // Unlock
	    atomicExch(e, slot);

	    return h;
	} else {
	    // The cell is unlocked and has a key in it, check if it matches

	    bool match = true;
	    for (int i = 0; i < kd && match; i++) {
		match = (table_keys[contents*kd+i] == key[i]);
	    }
	    if (match) return h;

	}
	// increment the bucket with wraparound
	h++;
	if (h == table_capacity*2) h = 0;
    }
}

template<int kd>
__device__ static int hashTableInsert(signed short *key, unsigned int slot) {
    unsigned int myHash = hash<kd>(key);
    return hashTableInsert<kd>(myHash, key, slot);
}



template<int kd> __device__ static
int hashTableRetrieveWithHash(unsigned int fh, signed short *key) {
  int h = modHash(fh);
  while (1) {
    int *e = table_entries + h;

    if (*e == -1) return -1;


    bool match = true;
    for (int i = 0; i < kd && match; i++) {
      match = (table_keys[(*e)*kd+i] == key[i]);
    }
    if (match) return *e;

    h++;
    if (h == table_capacity*2) h = 0;
  }
}

template<int kd>
__device__ static int hashTableRetrieve(signed short *key) {

    int h = modHash(hash<kd>(key));
    while (1) {
	int *e = table_entries + h;

	if (*e == -1) return -1;


	bool match = true;
	for (int i = 0; i < kd && match; i++) {
	    match = (table_keys[(*e)*kd+i] == key[i]);
	}
	if (match) return *e;

	h++;
	if (h == table_capacity*2) h = 0;
    }
}



//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// end of hash_table.cu



template<int pd>
__global__ static void createMatrix(const int w, const int h,
				    const float *positions,
				    const float *values,
				    const float *scaleFactor,
            int * matrix_int,
            float * matrix_float) {


    // 8x8 blocks
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    float myElevated[pd+1];
    const float *myPosition = positions + idx*pd;

    int myGreedy[pd+1];
    int myRank[pd+1];

    float myBarycentric[pd+2];
    __shared__ short keys[pd*BLOCK_SIZE];
    short *myKey = keys + threadId * pd;

    if (!outOfBounds) {

	  myElevated[pd] = -pd*(myPosition[pd-1])*scaleFactor[pd-1];
	  for (int i = pd-1; i > 0; i--) {
	    myElevated[i] = (myElevated[i+1] -
			     i*(myPosition[i-1])*scaleFactor[i-1] +
			     (i+2)*(myPosition[i])*scaleFactor[i]);
	  }
	  myElevated[0] = myElevated[1] + 2*(myPosition[0])*scaleFactor[0];

	  // find the closest zero-colored lattice point

	  // greedily search for the closest zero-colored lattice point
	  signed short sum = 0;
	  for (int i = 0; i <= pd; i++) {
	    float v = myElevated[i]*(1.0f/(pd+1));
	    float up = ceilf(v) * (pd+1);
	    float down = floorf(v) * (pd+1);
	    if (up - myElevated[i] < myElevated[i] - down) {
		    myGreedy[i] = (signed short)up;
	    } else {
		    myGreedy[i] = (signed short)down;
	    }
	    sum += myGreedy[i];
	  }
	  sum /= pd+1;

	  // sort differential to find the permutation between this simplex and the canonical one
	  for (int i = 0; i <= pd; i++) {
	    myRank[i] = 0;
	    for (int j = 0; j <= pd; j++) {
		    if (myElevated[i] - myGreedy[i] < myElevated[j] - myGreedy[j] ||
		    (myElevated[i] - myGreedy[i] == myElevated[j] - myGreedy[j]
		     && i > j)) {
		    myRank[i]++;
		    }
	    }
	  }


	if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
	    for (int i = 0; i <= pd; i++) {
		    if (myRank[i] >= pd + 1 - sum) {
		      myGreedy[i] -= pd+1;
		      myRank[i] += sum - (pd+1);
		      } else {
		        myRank[i] += sum;
		      }
	      }
	    } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
	      for (int i = 0; i <= pd; i++) {
		      if (myRank[i] < -sum) {
		        myGreedy[i] += pd+1;
		        myRank[i] += (pd+1) + sum;
		      } else {
		        myRank[i] += sum;
		      }
	      }
      }

  	// turn delta into barycentric coords
  	for (int i = 0; i <= pd+1; i++) {
  	    myBarycentric[i] = 0;
  	}

	  for (int i = 0; i <= pd; i++) {
	    float delta = (myElevated[i] - myGreedy[i]) * (1.0f/(pd+1));
	    myBarycentric[pd-myRank[i]] += delta;
	    myBarycentric[pd+1-myRank[i]] -= delta;
	   }
	  myBarycentric[0] += 1.0f + myBarycentric[pd+1];
  }

#ifdef USE_ADDITIVE_HASH
    unsigned int cumulative_hash = hash<pd>(myGreedy);
#endif
    for (int color = 0; color <= pd; color++) {
	// Compute the location of the lattice point explicitly (all but
	// the last coordinate - it's redundant because they sum to zero)
	if (!outOfBounds) {
	  for (int i = 0; i < pd; i++) {
		  myKey[i] = myGreedy[i] + color;
		    if (myRank[i] > pd-color) myKey[i] -= (pd+1);
	    }
	  }

#ifdef USE_ADDITIVE_HASH
	for (int i = 0; i < pd; i++) {
	    if (myRank[i] == pd-color) cumulative_hash += hOffset[i];
	}
#endif

  	if (!outOfBounds) {
        int r_index; // new
        float r_weight; // new

  	    #ifdef USE_ADDITIVE_HASH
        r_index = hashTableInsert<pd>(cumulative_hash, myKey, idx*(pd+1)+color); // new
  	    #else
        r_index = hashTableInsert<pd>(myKey, idx*(pd+1)+color); // new
  	    #endif
        r_weight = myBarycentric[color]; // new

        matrix_int[idx*(pd+1) + color] = r_index; // new
        matrix_float[idx*(pd+1) + color] = r_weight; // new


	  }
  }
}

template<int kd>
__global__ static void cleanHashTable(int n) {
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // find my hash table entry
    int *e = table_entries + idx;

    // Check if I created my own key in the previous phase
    if (*e >= 0) {
	// Rehash my key and reset the pointer in order to merge with
	// any other pixel that created a different entry under the
	// same key. If the computation was serial this would never
	// happen, but sometimes race conditions can make the same key
	// be inserted twice. hashTableRetrieve always returns the
	// earlier, so it's no problem as long as we rehash now.

	*e = hashTableRetrieve<kd>(table_keys + *e*kd);
    }
}



template<int pd, int vd>
__global__ static void splatCache(const int w, const int h, float *values, int * matrix_int,  float * matrix_float) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + (blockIdx.y/(pd+1)) * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int color = blockIdx.y % (pd+1);
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);


    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE*(vd+1)];
    int myOffset = -1;
    float *myValue = sharedValues + threadId*(vd+1);

    if (!outOfBounds) {

	  float *value = values + idx*vd;

    int r_index    =  matrix_int[idx*(pd+1)+color]; // new
    float r_weight =  matrix_float[idx*(pd+1)+color]; // new

	  // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
    matrix_int[idx*(pd+1)+color] = r_index = table_entries[r_index];// new

	  // record the offset into the keys/values array in shared space
    myOffset = sharedOffsets[threadId] = r_index*(vd+1); // new


	  for (int j = 0; j < vd; j++) {
       myValue[j] = value[j]*r_weight; // new

	  }
    myValue[vd] = r_weight; // new

    } else {
	    sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?

    if (outOfBounds) return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
	    if (i < threadId) {
	      if (myOffset == sharedOffsets[i]) {
		      // somebody else with higher priority cares about this key
		      return;
	      }
	  } else if (i > threadId) {
	    if (myOffset == sharedOffsets[i]) {
		  // someone else with lower priority cares about this key, accumulate it into mine
		    for (int j = 0; j <= vd; j++) {
		      sharedValues[threadId*(vd+1) + j] += sharedValues[i*(vd+1) + j];
		    }
	    }
	  }
  }

    // only the threads with something to write to main memory are still going
    float *val = table_values + myOffset;
    for (int j = 0; j <= vd; j++) {
	    atomicAdd(val+j, myValue[j]);
    }
  }

template<int pd, int vd>
__global__ static void blur(int n, float *newValues,int color, int * matrix_int, float * matrix_float, bool reverse) {

    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // Check if I'm valid
    if (matrix_int[idx] != idx) return; // new


    // find my key and the keys of my neighbours
    short myKey[pd+1];
    short np[pd+1];
    short nm[pd+1];

    for (int i = 0; i < pd; i++) {
        myKey[i] = table_keys[idx*pd+i];
	      np[i] = myKey[i]+1;
	      nm[i] = myKey[i]-1;
    }


    np[color] -= pd+1;
    nm[color] += pd+1;

#ifdef USE_ADDITIVE_HASH
    unsigned int hCurrent = hash<pd>(myKey);
    int offNp = hashTableRetrieveWithHash<pd>(hCurrent+hOffset[color],np);
    int offNm = hashTableRetrieveWithHash<pd>(hCurrent-hOffset[color],nm);
#else
    int offNp = hashTableRetrieve<pd>(np);
    int offNm = hashTableRetrieve<pd>(nm);
#endif

    float *valMe = table_values + (vd+1)*idx;
    float *valNp = table_values + (vd+1)*offNp;
    float *valNm = table_values + (vd+1)*offNm;
    float *valOut = newValues + (vd+1)*idx;

    if (offNp >= 0 && offNm >= 0) {
	    // for (int i = 0; i <= vd; i++) { // old
      for (int i = reverse?vd:0; i <= vd && i >= 0; reverse?i--:i++) { // new

	      valOut[i] = (valNp[i] + (valMe[i]*2) + valNm[i])/4;
	    }
    } else if (offNp >= 0) {
	    // for (int i = 0; i <= vd; i++) {// old
      for (int i = reverse?vd:0; i <= vd && i >= 0; reverse?i--:i++) { // new
	      valOut[i] = (valNp[i] + (valMe[i]*2))/4;
	    }
    } else if (offNm >= 0) {
	    // for (int i = 0; i <= vd; i++) {// old
      for (int i = reverse?vd:0; i <= vd && i >= 0; reverse?i--:i++) { // new
	      valOut[i] = (valNm[i] + (valMe[i]*2))/4;
	    }
    } else {
	    // for (int i = 0; i <= vd; i++) {// old
      for (int i = reverse?vd:0; i <= vd && i >= 0; reverse?i--:i++) { // new
	      valOut[i] = valMe[i]*2;
	    }
    }
  }

template<int pd, int vd>
__global__ static void slice(const int w, const int h, float *values, int * matrix_int, float * matrix_float) {


    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    if (outOfBounds) return;

    __shared__ float localValue[BLOCK_SIZE*vd];

    float *myValue = localValue + threadId*vd;
    float myWeight = 0;

    for (int i = 0; i < vd; i++) {
	    myValue[i] = 0;
    }

    for (int i = 0; i <= pd; i++) {
      int r_index = matrix_int[idx*(pd+1) + i]; // new
      float r_weight = matrix_float[idx*(pd+1) + i];// new

      float *val = table_values + r_index*(vd+1); // new

	    for (int j = 0; j < vd; j++) {
        myValue[j] += r_weight*val[j]; // new

	    }
	    myWeight += r_weight*val[vd];// new

    }

    myWeight = 1.0f/myWeight;
    for (int j = 0; j < vd; j++)
	    values[idx*vd + j] = myValue[j]*myWeight;
}


// assign scalars
__global__ void assignParams(const float * spat_f, const float * col_f, const bool * reverse_f, float * spat_d, float * col_d, bool * reverse_d) {
  spat_d[0] = spat_f[0];
  col_d[0] = col_f[0];
  reverse_d[0] = reverse_f[0];

}

__global__ static void fillRef(const float* ref_image, int w, int h, float spat , float col, float * ref ) {

  int i = 0;
  int j = 0;

  for (int y = 0; y < h; y++) {
     for (int x = 0; x < w; x++) {
        ref[i++] = spat * x;
        ref[i++] = spat * y;
  	    ref[i++] = col * ref_image[j++];
  	    ref[i++] = col * ref_image[j++];
  	    ref[i++] = col * ref_image[j++];
       }
  }
} // void fillRef


__global__ static void fillScaleFactor(float* scaleFactor, float blurVariance, int pd) {

  for (int i = 0; i < pd; i++) {
    scaleFactor[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
  }
} // void fillScaleFactor


__global__ static void copyKernel(const float* source, int N, float * dest ) {

  int tid=threadIdx.x+blockIdx.x*blockDim.x;

  if(tid<N){
      dest[tid]=source[tid];
  }
  // for (int i = 0; i < N; i++) {
  // 	    dest[i] = source[i];
  // }
} // void copyKernel

__global__ static void copyKernel(float* source, int N, float * dest ) {

  int tid=threadIdx.x+blockIdx.x*blockDim.x;

  // for (int c = 0; c < n; c++) {
  //     dest[c] = source[c];
  // }

  if(tid<N){
      dest[tid]=source[tid];
  }

} // void copyKernel



template<int vd, int pd>
void filter_(const float *input, const float * ref_image, float *output, int w, int h, int nChannels,const float * spat_f,const float * col_f,
  float * ref,
  float * values_out,
  float * newValues,
  float * scaleFactor,
  float * values_table,
    int * entries_table,
  short * keys_table,
  float * matrix_float,
    int * matrix_int,
    const bool * reverse_f) {

    printf("hello from void filter_(...)\n");

    CUT_CHECK_ERROR("CUDA error on function calling\n");


    size_t mem_tot_0, mem_free_0, mem_tot_1, mem_free_1;


    cudaMemGetInfo  (&mem_free_1, & mem_tot_1);
    printf("1 Free memory : %lu, Total memory : %lu\n",mem_free_1, mem_tot_1);


    float col[1],spat[1];
    bool reverse[1];
    float *col_d,*spat_d;
    bool * reverse_d;
    cudaMalloc(&col_d, sizeof(float));
    cudaMalloc(&spat_d, sizeof(float));
    cudaMalloc(&reverse_d, sizeof(bool));

    assignParams<<<1,1>>>(spat_f,col_f,reverse_f,spat_d,col_d,reverse_d);
    CUT_CHECK_ERROR("assignParams failed\n");


    cudaMemcpy(col, col_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spat, spat_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(reverse, reverse_d, sizeof(bool), cudaMemcpyDeviceToHost);


    float invColorStdev = 1.0 / col[0];
    float invSpatialStdev = 1.0 / spat[0];

    int nPixel = w*h; // width x height


    fillRef<<<1,1>>>(ref_image, w, h, invSpatialStdev , invColorStdev, ref);
    CUT_CHECK_ERROR("fillRef failed\n");


    float blurVariance = 0.5; // accurate setting
    fillScaleFactor<<<1,1>>>(scaleFactor,blurVariance,pd);
    CUT_CHECK_ERROR("scaleFactor failed\n");

    // Have to copy entries of "input" since it is declared const and we do not want that
    copyKernel<<<(vd*nPixel)/1024+1, 1024>>>(input, vd*nPixel, values_out); // new
    CUT_CHECK_ERROR("copyInputKernel failed\n");


    CUDA_SAFE_CALL(cudaMemset((void *)values_table, 0, nPixel*(pd+1)*pd*sizeof(float))); // why is nPixel*(pd+1)*vd not big enough?
    CUDA_SAFE_CALL(cudaMemset((void *)entries_table, -1, nPixel*(pd+1)*2*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset((void *)keys_table, 0, nPixel*(pd+1)*pd*sizeof(signed short)));


    createHashTable<pd, vd+1>(nPixel*(pd+1), values_table, entries_table, keys_table);
    CUT_CHECK_ERROR("createHashTable failed\n");


    // Populate constant memory for hash helpers
    unsigned long long int __host_two32 = ((unsigned long long int)1)<<32;
    unsigned int __host_div_c = 2*(nPixel*(pd+1));
    unsigned int __host_div_l = ceilf(logf((float)__host_div_c) / logf(2.0f));
    unsigned int __host_div_m = (__host_two32<<__host_div_l)/__host_div_c - __host_two32 + 1;


    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_m, &__host_div_m, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_l, &__host_div_l, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_c, &__host_div_c, sizeof(unsigned int)));
    CUT_CHECK_ERROR("__div_c failed\n");


    // Populate constant memory with hash of offset vectors
    unsigned int hOffset_host[pd+1];
    signed short offset[pd+1];
    for (int i = 0; i < pd; offset[i] = 1, i++);
    for (int i = 0; i <= pd; i++) {
      offset[i] -= pd+1; hOffset_host[i] = hash<pd>(offset); offset[i] += pd+1;
    }

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(hOffset, &hOffset_host, sizeof(unsigned int)*(pd+1)));
    CUT_CHECK_ERROR("hOffset failed\n");


    dim3 blocks((w-1)/8+1, (h-1)/8+1, 1);
    dim3 blockSize(8, 8, 1);

    timeval t[7];


    gettimeofday(t+0, NULL);

    createMatrix<pd><<<blocks, blockSize>>>(w, h, ref,values_out,scaleFactor, matrix_int, matrix_float);
    CUT_CHECK_ERROR("Matrix creation failed\n");

    gettimeofday(t+1, NULL);

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((nPixel-1)/cleanBlockSize+1, 2*(pd+1), 1);
    cleanHashTable<pd><<<cleanBlocks, cleanBlockSize>>>(2*nPixel*(pd+1));
    CUT_CHECK_ERROR("clean failed\n");



    gettimeofday(t+2, NULL);

    // splat splits by color, so extend the y coordinate to our blocks to represent that
    blocks.y *= pd+1;
    splatCache<pd, vd><<<blocks, blockSize>>>(w, h, values_out, matrix_int, matrix_float);
    CUT_CHECK_ERROR("splat failed\n");

    gettimeofday(t+3, NULL);

	  for (int color = 0; color <= pd; color++) {
	    blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(nPixel*(pd+1), newValues, color, matrix_int, matrix_float, reverse);
	    CUT_CHECK_ERROR("blur failed\n");
	    newValues = swapHashTableValues(newValues);
	  }



    gettimeofday(t+4, NULL);

    blocks.y /= (pd+1);
    slice<pd, vd><<<blocks, blockSize>>>(w, h, values_out, matrix_int, matrix_float);
    CUT_CHECK_ERROR("slice failed\n");

    gettimeofday(t+5, NULL);

    double total = (t[5].tv_sec - t[0].tv_sec)*1000.0 + (t[5].tv_usec - t[0].tv_usec)/1000.0;
    printf("Total time: %3.3f ms\n", total);

    char *names[5] = {"Create",
		      "Clean ",
		      "Splat ",
		      "Blur  ",
		      "Slice "};

    for (int i = 1; i < 6; i++) {
	    printf("%s: %3.3f ms\n", names[i-1], (t[i].tv_sec - t[i-1].tv_sec)*1000.0 + (t[i].tv_usec - t[i-1].tv_usec)/1000.0);
    }


    // copy input_h values from values to output
    copyKernel<<<(vd*nPixel)/1024+1, 1024>>>(values_out, vd*nPixel, output); // new
    CUT_CHECK_ERROR("writeOutputKernel did not work");


    printf("cleaning up CUDA memory\n" );
    // free allocated memory
    cudaFree(col_d);
    cudaFree(spat_d);
    // cudaFree(hOffset);// unfreed memory, but invalid device pointer error if not uncommented
    CUT_CHECK_ERROR("cleaning up did not work");


    cudaMemGetInfo (&mem_free_0, & mem_tot_0);
    printf("GPU free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);

    int diff = mem_free_0 - mem_free_1;

    printf("Difference between filter start and filter end: %i\n",diff );

}

#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif


void filter(const float *input, const float * ref_image, float *output, int pd, int vd, int w, int h, int nChannels,const float * spat_f,const  float * col_f,
  float * ref,
  float * values_out,
  float * newValues,
  float * scaleFactor,
  float * values_table,
    int * entries_table,
  short * keys_table,
  float * matrix_float,
    int * matrix_int,
    const bool * reverse) {

    switch (vd*1000 + pd) {
    case 1001: filter_<1, 1>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2001: filter_<2, 1>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3001: filter_<3, 1>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1002: filter_<1, 2>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2002: filter_<2, 2>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3002: filter_<3, 2>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1003: filter_<1, 3>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2003: filter_<2, 3>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3003: filter_<3, 3>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1004: filter_<1, 4>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2004: filter_<2, 4>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3004: filter_<3, 4>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1005: filter_<1, 5>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2005: filter_<2, 5>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3005: filter_<3, 5>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break; // this one
    case 1006: filter_<1, 6>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2006: filter_<2, 6>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3006: filter_<3, 6>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1007: filter_<1, 7>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2007: filter_<2, 7>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3007: filter_<3, 7>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1008: filter_<1, 8>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2008: filter_<2, 8>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3008: filter_<3, 8>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1009: filter_<1, 9>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2009: filter_<2, 9>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3009: filter_<3, 9>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1010: filter_<1, 10>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2010: filter_<2, 10>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3010: filter_<3, 10>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1011: filter_<1, 11>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2011: filter_<2, 11>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3011: filter_<3, 11>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1012: filter_<1, 12>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2012: filter_<2, 12>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3012: filter_<3, 12>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1013: filter_<1, 13>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2013: filter_<2, 13>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3013: filter_<3, 13>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1014: filter_<1, 14>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2014: filter_<2, 14>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3014: filter_<3, 14>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1015: filter_<1, 15>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2015: filter_<2, 15>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3015: filter_<3, 15>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 1016: filter_<1, 16>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 2016: filter_<2, 16>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 3016: filter_<3, 16>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break;
    case 4005: filter_<4, 5>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break; // new test case
    case 64005: filter_<64, 5>(input, ref_image, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, values_table, entries_table, keys_table, matrix_float, matrix_int, reverse); break; // case for Ole
    default:
	printf("Unsupported channel counts. Reference image must have 1 to 16 channels, input image must have 1 to 3 channels\n");
    }
}



#endif // if GOOGLE_CUDA
