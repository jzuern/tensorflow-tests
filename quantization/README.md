# 1: compile quantization Script


bazel build tensorflow/tools/quantization:quantize_graph


# 2: run quantization Script:

bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--output=input=tensorflow/examples/label_image/data/tensorflow_inception_graph_quantized.pb \
--output_node_names=final_result \
--mode=eightbit


# run model on image

bazel-bin/tensorflow/examples/label_image/label_image \
--graph=tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--image=tensorflow/examples/label_image/data/examples/book.jpg

# ToDo 24.01.

- run inception graph pb file on 10 images
- run quantized inception graph pb file on 10 images and compare results
- find out what modi "quantized", "8bit" , "weights", .. mean
  - round:           ?? (works not)
  - quantize:        ?? (works not)
  - eightbit:        converts 32bit floats to 8bit ints (works)
  - weights:         ?? (works)
  - weights_rounded: ?? (works)
