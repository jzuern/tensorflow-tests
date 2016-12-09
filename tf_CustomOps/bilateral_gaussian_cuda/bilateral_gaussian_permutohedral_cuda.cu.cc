

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#define BLOCK_SIZE 64
#define _DEBUG


#include "cutil.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_memory.h"
#include <sys/time.h>
#include "hash_table.cu"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"



#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif

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

__global__ static void makeRefKernel(const float* input, int w, int h, float spat , float col, float * ref_d ) {

  int i = 0;
  int j = 0;

  for (int y = 0; y < h; y++) {
     for (int x = 0; x < w; x++) {
        ref_d[i++] = spat * x;
        ref_d[i++] = spat * y;
  	    ref_d[i++] = col * input[j++];
  	    ref_d[i++] = col * input[j++];
  	    ref_d[i++] = col * input[j++];
       }
  }
} // void makeRefKernel


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

__global__ static void testKernel(float * test){

  test[0] = 3.32;
  // printf("test[0] = %f\n", test[0]);
}



template<int vd, int pd>
void filter_(const float *input, float *output, int w, int h, int nChannels,const float * spat_f,const float * col_f, bool accurate, float * test) {

    printf("hello from void filter_(...)\n");


    size_t mem_tot_0,mem_free_0;
    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("1 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);

    testKernel<<<1,1>>>(test);

    /* variables significant to memory usage:
    ref
    values
    matrix
    col_d,spat_d


    */


    float col[1],spat[1];
    float *col_d,*spat_d;
    cudaMalloc(&col_d, sizeof(float));
    cudaMalloc(&spat_d, sizeof(float));

    assignParams<<<1,1>>>(spat_f,col_f,spat_d,col_d);

    cudaMemcpy(col, col_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spat, spat_d, sizeof(float), cudaMemcpyDeviceToHost);


    float invColorStdev = 1.0 / col[0];
    float invSpatialStdev = 1.0 / spat[0];

    int n = w*h; // width x height

    float *ref; //ref array in device memory

    cudaMalloc(&ref, n*pd * sizeof(float));
    makeRefKernel<<<1,1>>>(input, w, h, invSpatialStdev , invColorStdev, ref);



    float blurVariance = accurate ? 0.5 : 0;

    float scaleFactor_h[pd];
    float *scaleFactor;
    for (int i = 0; i < pd; i++) {
      scaleFactor_h[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
    }
    cudaMalloc(&scaleFactor, pd * sizeof(float));
    cudaMemcpy(scaleFactor, scaleFactor_h, pd*sizeof(float), cudaMemcpyHostToDevice);

    printf("1.5 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    float * values;
    MatrixEntry * matrix;

    // copy stuff to device
    cudaMalloc(&values, vd*n*sizeof(float));
    cudaMalloc(&matrix, n*(pd+1)*sizeof(MatrixEntry));

    // Have to copy entries of "input" since it is declared const and we do not want that
    copyInputKernel<<<1,1>>>(input, vd*n, values);

    createHashTable<pd, vd+1>(n*(pd+1));

    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("2 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);




    // Populate constant memory for hash helpers
    unsigned long long int __host_two32 = ((unsigned long long int)1)<<32;
    unsigned int __host_div_c = 2*(n*(pd+1));
    unsigned int __host_div_l = ceilf(logf((float)__host_div_c) / logf(2.0f));
    unsigned int __host_div_m = (__host_two32<<__host_div_l)/__host_div_c - __host_two32 + 1;


    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_m, &__host_div_m, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_l, &__host_div_l, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(__div_c, &__host_div_c, sizeof(unsigned int)));


    // Populate constant memory with hash of offset vectors
    unsigned int hOffset_host[pd+1];
    signed short offset[pd+1];
    for (int i = 0; i < pd; offset[i] = 1, i++);
    for (int i = 0; i <= pd; i++) {
      offset[i] -= pd+1; hOffset_host[i] = hash<pd>(offset); offset[i] += pd+1;
    }

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(hOffset, &hOffset_host, sizeof(unsigned int)*(pd+1)));


    dim3 blocks((w-1)/8+1, (h-1)/8+1, 1);
    dim3 blockSize(8, 8, 1);

    timeval t[7];

    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("3 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);


    gettimeofday(t+0, NULL);

    createMatrix<pd><<<blocks, blockSize>>>(w, h, ref,values,scaleFactor,matrix); // allocate stuff in "matrix"
    CUT_CHECK_ERROR("Matrix creation failed\n");

    gettimeofday(t+1, NULL);

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((n-1)/cleanBlockSize+1, 2*(pd+1), 1);
    cleanHashTable<pd><<<cleanBlocks, cleanBlockSize>>>(2*n*(pd+1), matrix);

    CUT_CHECK_ERROR("clean failed\n");


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("4 Free memory : %lu, Total memory :%lu\n",mem_free_0, mem_tot_0);



    gettimeofday(t+2, NULL);

    // splat splits by color, so extend the y coordinate to our blocks to represent that
    blocks.y *= pd+1;
    splatCache<pd, vd><<<blocks, blockSize>>>(w, h, values, matrix);

    CUT_CHECK_ERROR("splat failed\n");

    gettimeofday(t+3, NULL);

    if (accurate) {
	    float *newValues;
	    allocateCudaMemory((void**)&(newValues), n*(pd+1)*(vd+1)*sizeof(float));
	    CUDA_SAFE_CALL(cudaMemset((void *)newValues, 0, n*(pd+1)*(vd+1)*sizeof(float)));

	    for (int color = 0; color <= pd; color++) {
	      blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(n*(pd+1), newValues, matrix, color);
	      CUT_CHECK_ERROR("blur failed\n");
	      newValues = swapHashTableValues(newValues);
	    }
    }

    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("5 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);



    gettimeofday(t+4, NULL);

    blocks.y /= (pd+1);
    slice<pd, vd><<<blocks, blockSize>>>(w, h, values, matrix);

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
    printf("Total GPU memory usage: %u bytes\n", (unsigned int)GPU_MEMORY_ALLOCATION);


    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("6 Free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);

    printf("cleaning up CUDA memory\n" );

    destroyHashTable();

    // copy input_h values from values to output
    writeOutputKernel<<<1,1>>>(values, vd*n, output);

    cudaFree(values);
    cudaFree(matrix);
    cudaFree(ref);



    cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
    printf("---GPU free memory : %lu, Total memory : %lu\n",mem_free_0, mem_tot_0);

}

#ifdef LIBRARY
extern "C"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif


void filter(const float *input, float *output, int pd, int vd, int w, int h, int nChannels,const float * spat_f,const  float * col_f, bool accurate, float * test) {

    switch (vd*1000 + pd) {
    case 1001: filter_<1, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2001: filter_<2, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3001: filter_<3, 1>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1002: filter_<1, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2002: filter_<2, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3002: filter_<3, 2>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1003: filter_<1, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2003: filter_<2, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3003: filter_<3, 3>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1004: filter_<1, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2004: filter_<2, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3004: filter_<3, 4>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1005: filter_<1, 5>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2005: filter_<2, 5>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3005: filter_<3, 5>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1006: filter_<1, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2006: filter_<2, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3006: filter_<3, 6>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1007: filter_<1, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2007: filter_<2, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3007: filter_<3, 7>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1008: filter_<1, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2008: filter_<2, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3008: filter_<3, 8>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1009: filter_<1, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2009: filter_<2, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3009: filter_<3, 9>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1010: filter_<1, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2010: filter_<2, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3010: filter_<3, 10>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1011: filter_<1, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2011: filter_<2, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3011: filter_<3, 11>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1012: filter_<1, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2012: filter_<2, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3012: filter_<3, 12>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1013: filter_<1, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2013: filter_<2, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3013: filter_<3, 13>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1014: filter_<1, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2014: filter_<2, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3014: filter_<3, 14>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1015: filter_<1, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2015: filter_<2, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3015: filter_<3, 15>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 1016: filter_<1, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 2016: filter_<2, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    case 3016: filter_<3, 16>(input, output, w, h, nChannels, spat_f, col_f, accurate, test); break;
    default:
	printf("Unsupported channel counts. Reference image must have 1 to 16 channels, input image must have 1 to 3 channels\n");
    }
}



#endif // if GOOGLE_CUDA
