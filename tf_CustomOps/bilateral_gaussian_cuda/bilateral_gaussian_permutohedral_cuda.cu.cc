

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

__device__ __constant__ unsigned int hOffset[64]; // issue

// allocate hash table pointers on global scope
float *values;
int *entries;
signed short *keys;

template<int kd, int vd>
void createHashTable(int capacity) {

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_capacity,
				      &capacity,
				      sizeof(unsigned int)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&values, capacity*vd*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset((void *)values, 0, capacity*vd*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
				      &values,
				      sizeof(float *)));



    CUDA_SAFE_CALL(cudaMalloc((void **)&entries, capacity*2*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset((void *)entries, -1, capacity*2*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_entries,
				      &entries,
				      sizeof(unsigned int *)));



    CUDA_SAFE_CALL(cudaMalloc((void **)&keys, capacity*kd*sizeof(signed short)));
    CUDA_SAFE_CALL(cudaMemset((void *)keys, 0, capacity*kd*sizeof(signed short)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_keys,
				      &keys,
				      sizeof(unsigned int *)));

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



struct MatrixEntry {
    int index;
    float weight;
};



template<int pd>
__global__ static void createMatrix(const int w, const int h,
				    const float *positions,
				    const float *values,
				    const float *scaleFactor,
				    MatrixEntry *matrix) {


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


#ifdef LINEAR_D_MEMORY
	    for (int i = 0; i <= pd; i++) {
	      table_zeros[idx*(pd+1)+i] = myGreedy[i];
	      table_rank[idx*(pd+1)+i] = myRank[i];
	    }
#endif

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
  	    MatrixEntry r;
  	    #ifdef USE_ADDITIVE_HASH
  	    r.index = hashTableInsert<pd>(cumulative_hash, myKey, idx*(pd+1)+color);
  	    #else
  	    r.index = hashTableInsert<pd>(myKey, idx*(pd+1)+color);
  	    #endif
  	    r.weight = myBarycentric[color];
  	    matrix[idx*(pd+1) + color] = r;
	  }
  }
}

template<int kd>
__global__ static void cleanHashTable(int n, MatrixEntry *matrix) {
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

#ifdef LINEAR_D_MEMORY
  // Get my key
  short myKey[kd];
  generateKey<kd>(*e, myKey);
	*e = hashTableRetrieve<kd>(myKey);
#else
	*e = hashTableRetrieve<kd>(table_keys + *e*kd);
#endif
    }
}



template<int pd, int vd>
__global__ static void splatCache(const int w, const int h, float *values, MatrixEntry *matrix) {

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

	  MatrixEntry r = matrix[idx*(pd+1)+color];

	  // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
	  matrix[idx*(pd+1)+color].index = r.index = table_entries[r.index];
	  // record the offset into the keys/values array in shared space
	  myOffset = sharedOffsets[threadId] = r.index*(vd+1);

	  for (int j = 0; j < vd; j++) {
	     myValue[j] = value[j]*r.weight;
	  }
	  myValue[vd] = r.weight;
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
__global__ static void blur(int n, float *newValues, MatrixEntry *matrix, int color) {

    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // Check if I'm valid
    if (matrix[idx].index != idx) return;


    // find my key and the keys of my neighbours
    short myKey[pd+1];
    short np[pd+1];
    short nm[pd+1];

    #ifdef LINEAR_D_MEMORY
    generateKey<pd>(idx, myKey);
    for (int i = 0; i < pd; i++) {
	    np[i] = myKey[i]+1;
	    nm[i] = myKey[i]-1;
    }
    #else
    for (int i = 0; i < pd; i++) {
        myKey[i] = table_keys[idx*pd+i];
	      np[i] = myKey[i]+1;
	      nm[i] = myKey[i]-1;
    }
    #endif

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
	    for (int i = 0; i <= vd; i++) {
	      valOut[i] = (valNp[i] + (valMe[i]*2) + valNm[i])/4;
	    }
    } else if (offNp >= 0) {
	    for (int i = 0; i <= vd; i++) {
	      valOut[i] = (valNp[i] + (valMe[i]*2))/4;
	    }
    } else if (offNm >= 0) {
	    for (int i = 0; i <= vd; i++) {
	      valOut[i] = (valNm[i] + (valMe[i]*2))/4;
	    }
    } else {
	    for (int i = 0; i <= vd; i++) {
	      valOut[i] = valMe[i]*2;
	    }
    }
  }

template<int pd, int vd>
__global__ static void slice(const int w, const int h, float *values, MatrixEntry *matrix) {

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
	    MatrixEntry r = matrix[idx*(pd+1) + i];
	    float *val = table_values + r.index*(vd+1);
	    for (int j = 0; j < vd; j++) {
	      myValue[j] += r.weight*val[j];
	    }
	    myWeight += r.weight*val[vd];
    }

    myWeight = 1.0f/myWeight;
    for (int j = 0; j < vd; j++)
	    values[idx*vd + j] = myValue[j]*myWeight;
}


// assign scalar floats
__global__ void assignParams(const float * spat_f, const float * col_f, float * spat_d, float * col_d) {
  spat_d[0] = spat_f[0];
  col_d[0] = col_f[0];
}

__global__ static void fillRef(const float* input, int w, int h, float spat , float col, float * ref ) {

  int i = 0;
  int j = 0;

  for (int y = 0; y < h; y++) {
     for (int x = 0; x < w; x++) {
        ref[i++] = spat * x;
        ref[i++] = spat * y;
  	    ref[i++] = col * input[j++];
  	    ref[i++] = col * input[j++];
  	    ref[i++] = col * input[j++];
       }
  }
} // void fillRef


__global__ static void fillScaleFactor(float* scaleFactor, float blurVariance, int pd) {

  for (int i = 0; i < pd; i++) {
    scaleFactor[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
  }
} // void fillScaleFactor



__global__ static void copyInputKernel(const float* input, int N, float * values ) {

  for (int i = 0; i < N; i++) {
  	    values[i] = input[i];
  }
} // void copyInputKernel

__global__ static void writeOutputKernel(float* inp_h, int n, float * output ) {

  for (int c = 0; c < n; c++) {
      output[c] = inp_h[c];
  }
} // void writeOutputKernel

__global__ static void testKernel(float * tensor){

  tensor[0] = 1.0;
  printf("tensor[0] = %f\n", tensor[0]);
}

__global__ static void setZeroKernel(float* newValues, int N) {

  for (int i = 0; i < N; i++) {
        newValues[i] = 0.0;
  }
} // void setZeroKernel



template<int vd, int pd>
void filter_(const float *input, float *output, int w, int h, int nChannels,const float * spat_f,const float * col_f,
  float * ref,
  float * values_out,
  float * newValues,
  float * scaleFactor,
  float * table_test_ext,
  float * persistent) {

    printf("hello from void filter_(...)\n");

    // testKernel<<<1,1>>>(persistent); // why not working?
    CUT_CHECK_ERROR("testKernel failed\n");


    size_t mem_tot_0, mem_free_0, mem_tot_1, mem_free_1;


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("1 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    /* variables significant to memory usage:
    ref                                      .. to implement
    values_out                               .. to implement
    newValues                                .. to implement
    scaleFactor                              .. to implement

    matrix                                   .. allocated with cudaMalloc
    col_d,spat_d                             .. allocated with cudaMalloc
    table_entries,table_values,table_keys:   .. allocated with cudaMalloc
    hoffset                                  .. allocated with cudaMalloc
    */



    float col[1],spat[1];
    float *col_d,*spat_d;
    cudaMalloc(&col_d, sizeof(float));
    cudaMalloc(&spat_d, sizeof(float));

    assignParams<<<1,1>>>(spat_f,col_f,spat_d,col_d);
    CUT_CHECK_ERROR("assignParams failed\n");


    cudaMemcpy(col, col_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spat, spat_d, sizeof(float), cudaMemcpyDeviceToHost);


    float invColorStdev = 1.0 / col[0];
    float invSpatialStdev = 1.0 / spat[0];

    int nPixel = w*h; // width x height

    // float *ref; //ref array in device memory
    // cudaMalloc(&ref, nPixel*pd * sizeof(float));

    fillRef<<<1,1>>>(input, w, h, invSpatialStdev , invColorStdev, ref);
    CUT_CHECK_ERROR("fillRef failed\n");



    float blurVariance = 0.5; // accurate setting
    fillScaleFactor<<<1,1>>>(scaleFactor,blurVariance,pd);
    CUT_CHECK_ERROR("scaleFactor failed\n");

    // float scaleFactor_h[pd];
    // float *scaleFactor;
    // for (int i = 0; i < pd; i++) {
    //   scaleFactor_h[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
    // }
    // cudaMalloc(&scaleFactor, pd * sizeof(float));
    // cudaMemcpy(scaleFactor, scaleFactor_h, pd*sizeof(float), cudaMemcpyHostToDevice);


    printf("1.5 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    // float * values_out;
    MatrixEntry * matrix;

    // copy stuff to device
    // cudaMalloc(&values_out, vd*nPixel*sizeof(float));
    cudaMalloc(&matrix, nPixel*(pd+1)*sizeof(MatrixEntry));

    // Have to copy entries of "input" since it is declared const and we do not want that
    copyInputKernel<<<1,1>>>(input, vd*nPixel, values_out);
    CUT_CHECK_ERROR("copyInputKernel failed\n");


    createHashTable<pd, vd+1>(nPixel*(pd+1));
    CUT_CHECK_ERROR("createHashTable failed\n");

    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("2 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);




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

    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("3 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    gettimeofday(t+0, NULL);

    createMatrix<pd><<<blocks, blockSize>>>(w, h, ref,values_out,scaleFactor,matrix); // allocate stuff in "matrix"
    CUT_CHECK_ERROR("Matrix creation failed\n");

    gettimeofday(t+1, NULL);

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((nPixel-1)/cleanBlockSize+1, 2*(pd+1), 1);
    cleanHashTable<pd><<<cleanBlocks, cleanBlockSize>>>(2*nPixel*(pd+1), matrix);

    CUT_CHECK_ERROR("clean failed\n");


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("4 Free memory : %lu, Total memory :%lu\n",mem_free_0, mem_tot_0);



    gettimeofday(t+2, NULL);

    // splat splits by color, so extend the y coordinate to our blocks to represent that
    blocks.y *= pd+1;
    splatCache<pd, vd><<<blocks, blockSize>>>(w, h, values_out, matrix);

    CUT_CHECK_ERROR("splat failed\n");

    gettimeofday(t+3, NULL);

	  // float *newValues;
	  // CUDA_SAFE_CALL(cudaMalloc((void**)&(newValues), nPixel*(pd+1)*(vd+1)*sizeof(float)));
	  // CUDA_SAFE_CALL(cudaMemset((void *)newValues, 0, nPixel*(pd+1)*(vd+1)*sizeof(float)));

	  for (int color = 0; color <= pd; color++) {
	    blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(nPixel*(pd+1), newValues, matrix, color);
	    CUT_CHECK_ERROR("blur failed\n");
	    newValues = swapHashTableValues(newValues);
	  }


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("5 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);



    gettimeofday(t+4, NULL);

    blocks.y /= (pd+1);
    slice<pd, vd><<<blocks, blockSize>>>(w, h, values_out, matrix);

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


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("6 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    // copy input_h values from values to output
    writeOutputKernel<<<1,1>>>(values_out, vd*nPixel, output);
    CUT_CHECK_ERROR("writeOutputKernel up did not work");


    printf("cleaning up CUDAA memory\n" );

    // free allocated memory
    cudaFree(matrix);
    cudaFree(col_d);
    cudaFree(spat_d);
    // cudaFree(hOffset);// unfreed memory, but invalid device pointer error if not uncommented

    // free hash table stuff
    cudaFree(table_values);
    cudaFree(table_entries);
    cudaFree(table_keys);
    cudaFree(values);
    cudaFree(entries);
    cudaFree(keys);


    CUT_CHECK_ERROR("cleaning up did not work");



    cudaMemGetInfo (&mem_free_1, & mem_tot_1);
    printf("GPU free memory : %lu, Total memory : %lu\n",mem_free_1, mem_tot_1);

    int diff = mem_free_0 - mem_free_1;

    printf("Difference between filter start and filter end: %i\n",diff );

}

#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif


void filter(const float *input, float *output, int pd, int vd, int w, int h, int nChannels,const float * spat_f,const  float * col_f,
  float * ref,
  float * values_out,
  float * newValues,
  float * scaleFactor,
  float * table_test,
  float * pers) {

    switch (vd*1000 + pd) {
    // case 1001: filter_<1, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2001: filter_<2, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3001: filter_<3, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1002: filter_<1, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2002: filter_<2, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3002: filter_<3, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1003: filter_<1, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2003: filter_<2, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3003: filter_<3, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1004: filter_<1, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2004: filter_<2, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3004: filter_<3, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1005: filter_<1, 5>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2005: filter_<2, 5>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3005: filter_<3, 5>(input, output, w, h, nChannels, spat_f, col_f, ref, values_out, newValues, scaleFactor, table_test, pers); break;
    // case 1006: filter_<1, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2006: filter_<2, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3006: filter_<3, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1007: filter_<1, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2007: filter_<2, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3007: filter_<3, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1008: filter_<1, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2008: filter_<2, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3008: filter_<3, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1009: filter_<1, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2009: filter_<2, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3009: filter_<3, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1010: filter_<1, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2010: filter_<2, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3010: filter_<3, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1011: filter_<1, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2011: filter_<2, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3011: filter_<3, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1012: filter_<1, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2012: filter_<2, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3012: filter_<3, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1013: filter_<1, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2013: filter_<2, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3013: filter_<3, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1014: filter_<1, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2014: filter_<2, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3014: filter_<3, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1015: filter_<1, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2015: filter_<2, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3015: filter_<3, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 1016: filter_<1, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 2016: filter_<2, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    // case 3016: filter_<3, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    default:
	printf("Unsupported channel counts. Reference image must have 1 to 16 channels, input image must have 1 to 3 channels\n");
    }
}



#endif // if GOOGLE_CUDA
