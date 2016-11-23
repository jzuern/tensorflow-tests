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
    name = "bilateral_gaussian_permutohedral.so",
    srcs = ["bilateral_gaussian_permutohedral.cc"],

)
