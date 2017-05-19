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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void kernel(const float* test_in, const int N, float* test_out, const int dilationH, const int dilationW,
                                const int dH, const int dW, const int in_height, const int in_width, const int nBatch, const int nChannels,
                                const int out_height, const int out_width, const int padH, const int padW, const int kH, const int kW){

  printf("Hello from thread %d\n", threadIdx.x);



  const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
  if (idx >= N) return;



printf("PoolParameters: \n\nin_height = %i, in_width = %i,  \n"
                                  "window_rows = %i, window_cols = %i \n"
                                  "row_stride = %i, col_stride = %i, \n"
                                  "out_height = %i, out_width = %i, \n",
                                  in_height, in_width, kH ,kW , dH,  dW, out_height,out_width );



for (int b = 0; b < nBatch; b++) { // batch
  for (int c = 0; c < nChannels; c++) { // channel
    for (int i = 0; i < out_height; i++) { // heigth of output image
      for (int j = 0; j < out_width; j++) { // width of output image

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
            // float val = test_in(b,y,x,c);
            int idx1 = nChannels*in_height*in_width;
            int idx2 = in_height*in_width;
            int idx3 = in_width;
            int idx = b*idx1 + c * idx2 + y*idx3 + x;
            float val = test_in[idx];

            if (val > maxval) maxval = val;
          }
        }

        /* set output to local max */
        // test_out(b,i,j,c) = maxval;
        int idx1 = nChannels*out_height*out_width;
        int idx2 = out_height*out_width;
        int idx3 = out_width;
        int idx = b*idx1 + c * idx2 + i*idx3 + j;

        test_out[idx] = maxval;

      }
    }
  }
}

// print input
printf("input image: \n");

for(int i = 0; i < in_height; i++){
  for(int j = 0; j < in_width; j++){
    printf("%f ", test_in[i*in_width + j]);
  }
  printf("\n");
}
printf("output image: \n");

for(int i = 0; i < out_height; i++){
  for(int j = 0; j < out_width; j++){
    printf("%f ", test_out[i*out_width + j]);
  }
  printf("\n");
}




  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;i += blockDim.x * gridDim.x) {
  //   test_out[i] = test_in[i] + 1.0;
  // }
}

void DilatedMaxPoolingKernelLauncher(const float* test_in, const int N, float* test_out, const int dilationH, const int dilationW,
                                const int dH, const int dW, const int in_height, const int in_width, const int nBatch, const int nChannels,
                                const int out_height, const int out_width, const int padH, const int padW, const int kH, const int kW){
  kernel<<<1, 1>>>(test_in,N,test_out,dilationH,dilationW,dH,dW,in_height,in_width,nBatch,nChannels,out_height,out_width,padH,padW,kH,kW); // 32,256
}

#endif
