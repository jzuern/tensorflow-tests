# 1: compile quantization Script
bazel build tensorflow/tools/quantization:quantize_graph

# 2: run quantization Script:
bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--output=tensorflow/examples/label_image/data/tensorflow_inception_graph_eightbit_16bit.pb \
--output_node_names=softmax \
--mode=eightbit

# 3: run model on image 1
bazel-bin/tensorflow/examples/label_image/label_image \
--graph=tensorflow/examples/label_image/data/tensorflow_inception_graph_eightbit.pb \
--image=tensorflow/examples/label_image/data/examples/car.jpg

# 3: run model on image 2
bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph.pb \
--image=tensorflow/examples/label_image/data/examples/car.jpg

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#shrinking-file-size:

bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--out_graph=tensorflow/examples/label_image/data/tensorflow_inception_graph_testv1.pb \
--inputs='Mul:0' --outputs='softmax:0' --transforms='quantize_weights'


# freeze_graph test

bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/home/jzuern/tensorflow/tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--output_graph=/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph.pb \

#######################################################################################################


------------------------------

 export  CUDA_VISIBLE_DEVICES="" for non-GPU calculations


TODO 14.03.:

- Performance-Messung und Ergebnisgüte bei 5 Bildern, mit und ohne GPU
