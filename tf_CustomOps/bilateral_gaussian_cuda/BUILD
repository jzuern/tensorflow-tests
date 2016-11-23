
licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
    name = "bilateral_gaussian_permutohedral_cuda.so",
    srcs = ["bilateral_gaussian_permutohedral_cuda.cc"],
    gpu_srcs = ["bilateral_gaussian_permutohedral_cuda.cu.cc"],
    hdrs = ["cutil.h",
            "cuda_memory.h",
            "hash_table.cu",
            "MirroredArray.h",
            ],
)
