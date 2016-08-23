// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

/*#include <stdlib.h> // for sleep(uint)
*/

template<int DataLayout>
static void test_full_reductions() {

  printf("\n\n\n\n ---- Testing full reductions (CPU) ----\n\n\n\n");


  // set number of rows and columns of input tensor
  const int N = 64;

  // initialize input tensor
  Tensor<float, 2, DataLayout> in(N, N);
  in.setRandom();

  // initialize output tensor
  Tensor<float, 0, DataLayout> full_redux;


  // perform full sum reduction on CPU
  full_redux = in.sum();


  printf("\n\n\n\n ---- Testing full reductions (GPU) ----\n\n\n\n");


  // CUDA stuff
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  // data type sizes
  std::size_t in_bytes = in.size() * sizeof(float);
  std::size_t out_bytes = full_redux.size() * sizeof(float);

  // allocate floats on GPU
  float* gpu_in_ptr = static_cast<float*>(gpu_device.allocate(in_bytes));
  float* gpu_out_ptr = static_cast<float*>(gpu_device.allocate(out_bytes));

  // copy input tensor data from host to device
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);


  TensorMap<Tensor<float, 2, DataLayout> > in_gpu(gpu_in_ptr, N, N);
  TensorMap<Tensor<float, 0, DataLayout> > out_gpu(gpu_out_ptr);


  // perform full sum reduction on GPU (device)
  out_gpu.device(gpu_device) = in_gpu.sum();

  // initialize output tensor for gpu computation
  Tensor<float, 0, DataLayout> full_redux_gpu;

  // copy memory from device to host
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);

  // synchronize  (does What?)
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());

  // cleanup memory
  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}


static void test_partial_reductions(){ // jzuern partial reduction 

  printf("\n\n\n\n ---- Testing partial reductions (CPU) ----\n\n\n\n");


  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1000*1024*1024); // hundred megs output buffer


  // set number of rows and columns of input tensor
  const int N = 8;

  // initialize input tensor
  Tensor<float, 4, ColMajor> in(N, N, N, N);
  in.setRandom();

  // initialize output tensor
  Tensor<float, 1, ColMajor> part_redux(N);

  // define eigen array of dimensions to reduce along
  Eigen::array<int, 3> dims({1,2,3}); // 0,1,2 works. 1,2,3 does not
  // 0th dimension is innermost
  // nth dimension is outermost

  // perform partial sum reduction on CPU
  part_redux = in.sum(dims);


  printf("\n\n\n\n ---- Testing partial reductions (GPU) ----\n\n\n\n");

  // CUDA stuff
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  // data type sizes
  std::size_t in_bytes = in.size() * sizeof(float);
  std::size_t out_bytes = part_redux.size() * sizeof(float);

  // allocate floats on GPU
  float* gpu_in_ptr = static_cast<float*>(gpu_device.allocate(in_bytes));
  float* gpu_out_ptr = static_cast<float*>(gpu_device.allocate(out_bytes));

  // copy input tensor data from host to device
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<float, 4, ColMajor> > in_gpu (gpu_in_ptr, N, N, N, N);
  TensorMap<Tensor<float, 1, ColMajor> > out_gpu(gpu_out_ptr,N);


  // perform partial sum reduction on GPU (device)
  out_gpu.device(gpu_device) = in_gpu.sum(dims);


  // initialize output tensor for gpu computation
  Tensor<float, 1, ColMajor> part_redux_gpu(N);

  // copy memory from device to host
  gpu_device.memcpyDeviceToHost(part_redux_gpu.data(), gpu_out_ptr, out_bytes); // version 1
  //assert(cudaMemcpyAsync(part_redux.data(), gpu_out_ptr, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess); //version 2


  // synchronize  (does What?)
  gpu_device.synchronize(); // version 1
  //assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess); // version 2


  // cleanup memory
  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}


void test_cuda_reduction_steiner()
{
  Tensor<float, 4> in1(23,6,97,5);
  Tensor<float, 2> out(97,5);
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 23,6,97,5);
  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 97,5);

  array<int, 2> reduction_axis;
  reduction_axis[0] = 0;
  reduction_axis[1] = 1;

  gpu_out.device(gpu_device) = gpu_in1.sum(reduction_axis);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);


  cudaFree(d_in1);
  cudaFree(d_out);
}

void test_cxx11_tensor_reduction_cuda() {

  //CALL_SUBTEST(test_full_reductions<ColMajor>());
  CALL_SUBTEST(test_partial_reductions());
  //CALL_SUBTEST(test_cuda_reduction_steiner());
}
