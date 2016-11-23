# Description:
# Code examples referenced by adding_an_op

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")



tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)


tf_custom_op_library(
    name = "bilateral_gaussian_permutohedral_cuda.so",
    srcs = ["bilateral_gaussian_permutohedral_cuda.cc"],
    gpu_srcs = ["bilateral_gaussian_permutohedral_cuda.cu.cc"],
)
