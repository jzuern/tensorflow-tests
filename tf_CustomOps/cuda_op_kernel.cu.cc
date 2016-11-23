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

__global__ void AddOneKernel(const int* in, const int N, int* out) {


  // printf("Hello from Block %i, thread %i ",blockIdx.x,blockIdx.x);
  for (int i = 0; i < N; i++) {
    printf("in[i] == %i\n", in[i] ); // works just fine
    out[i] = in[i] + 1;
  }

  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
  //   printf("in[i] == %i\n", in[i] ); // works just fine
  //   out[i] = in[i] + 1;
  // }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {
  printf("hello from AddOneKernelLauncher\n");

  // printf("in[0] == %i\n\n\n", in[0]); // causes segfault

  // AddOneKernel<<<32, 2>>>(in, N, out);
  AddOneKernel<<<1, 1>>>(in, N, out);

}

#endif
