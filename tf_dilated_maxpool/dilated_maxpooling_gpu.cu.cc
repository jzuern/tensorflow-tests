
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


__global__ void dilated_maxpool_kernel(const float* tensor_in, const int N, float* tensor_out, const int dilationH, const int dilationW,
                                const int dH, const int dW, const int in_height, const int in_width, const int nBatch, const int nChannels,
                                const int out_height, const int out_width, const int padH, const int padW, const int kH, const int kW){


  // window_rows: kH
  // window_cols: kW

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; // width
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; // height
    // i : height
    // j : width

    // int idx1_in = nChannels*in_height*in_width;
    // int idx2_in = in_height*in_width;
    // int idx3_in = in_width;
    //
    // int idx1_out = nChannels*out_height*out_width;
    // int idx2_out = out_height*out_width;
    // int idx3_out = out_width;

    int idx1_in = in_height*in_width*nChannels;
    int idx2_in = in_width*nChannels;
    int idx3_in = nChannels;

    int idx1_out = out_height*out_width*nChannels;
    int idx2_out = out_width*nChannels;
    int idx3_out = nChannels;


    //Make sure the current thread is inside the image bounds
    if(xIndex<out_width && yIndex<out_height){
    //   printf("PoolParameters: \n\nin_height = %i, in_width = %i,  \n"
    //                                   "window_rows = %i, window_cols = %i \n"
    //                                   "row_stride = %i, col_stride = %i, \n"
    //                                   "out_height = %i, out_width = %i, \n",
    //                                   in_height, in_width, kH ,kW , dH,  dW, out_height,out_width );

          // printf("Hello from Thread with xIndex = %i, yIndex = %i\n ", xIndex,yIndex);

            int b = 0; // current batch
            for (int c = 0; c < nChannels; c++) { // channel

              int i = yIndex;
              int j = xIndex;

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

                  // int idx = b*idx1_in + c * idx2_in + y*idx3_in + x;
                  int idx = b*idx1_in + y * idx2_in + x*idx3_in + c;

                  float val = tensor_in[idx];

                  if (val > maxval) maxval = val;
                }
              }

            /* set output to local max */
            // int idx = b*idx1_out + c * idx2_out + i*idx3_out + j;
            int idx = b*idx1_out + i * idx2_out + j*idx3_out + c;

            tensor_out[idx] = maxval;
          }
        // } // batch entry loop
      }// if !outofbounds

}

void DilatedMaxPoolingKernelLauncher(const float* tensor_in, const int N, float* tensor_out, const int dilationH, const int dilationW,
                                const int dH, const int dW, const int in_height, const int in_width, const int nBatch, const int nChannels,
                                const int out_height, const int out_width, const int padH, const int padW, const int kH, const int kW){


    /*
    * Specify a block size. 256 threads per block are sufficient.
    * It can be increased, but keep in mind the limitations of the GPU.
    * Older GPUs allow maximum 512 threads per block.
    * Current GPUs allow maximum 1024 threads per block
    */

    dim3 threadsPerBlock(16,16);  //16*16 = 256

    /*
     * Specify the grid size for the GPU.
     * Make it generalized, so that the size of grid changes according to the input image size
     */

    dim3 blocksPerGrid;
    blocksPerGrid.x = (out_width + threadsPerBlock.x - 1)/threadsPerBlock.x;  /*< Greater than or equal to image width */
    blocksPerGrid.y = (out_height + threadsPerBlock.y - 1)/threadsPerBlock.y; /*< Greater than or equal to image height */

    printf("Kernel info: \n blocksPerGrid.x = %i \n blocksPerGrid.y = %i \n threadsPerBlock.x = %i \n threadsPerBlock.y = %i\n",
                          blocksPerGrid.x,blocksPerGrid.y,threadsPerBlock.x,threadsPerBlock.y);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  float * batch_pointer_in = (float *)tensor_in; // pointer to current batch entry for input image
  float * batch_pointer_out = tensor_out; // pointer to current batch entry for output image

  // outer loop through batch entries:
  for (int b = 0; b < nBatch; b++){
    printf("Launching kernel with batch entry %i\n",b);
    dilated_maxpool_kernel<<<blocksPerGrid, threadsPerBlock>>>(batch_pointer_in,N,batch_pointer_out,dilationH,dilationW,dH,dW,in_height,in_width,nBatch,nChannels,out_height,out_width,padH,padW,kH,kW); // 32,256
    batch_pointer_in +=  nChannels*in_height*in_width;
    batch_pointer_out +=  nChannels*out_height*out_width;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Executing all kernels took %f miliseconds\n", milliseconds);

}

#endif // CUDA
