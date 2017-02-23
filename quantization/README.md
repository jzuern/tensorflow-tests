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

## Modi

  - round (works not):           
    rounds output nodes

  - quantize (works not):       
    quantizes output nodes

  - eightbit (works):        
    converts 32bit floats to 8bit ints

  - weights (works):       
    float Const ops are quantized and replaced by a tuple of four ops to perform
    the dequantization at runtime:
    * eight-bit Const (bucket indices, same shape as original float Const op
    * two float Const ops (min and max value of original float Const op)
    * Dequantize op to convert the eight-bit consts to float tensors.
    The quantization mode is important because we see accuracy problems when
    quantizing weights for different situations depending on the algorithm
    used. We haven't figured out exactly what the underlying cause is yet,
    unfortunately.

  - weights_rounded (works):
    this function replaces float Const ops with quantized float Const ops - same as the original op, but
    float values being mapped to the center of one of 1<<FLAGS.bitdepth buckets.
    1 << bitdepth == 1* 2 ^bitdepth
    --> This does not change the raw model size,
    but compression algorithms such as zip (as used for compressing apks) or bzip2 will achieve a very good compression ratio




# 04.02.:

- Ergebnisse:
  - 8bit/16bit hat Einfluss auf Ergebnis bei:     mode=weights_rounded
  - 8bit/16bit hat Einfluss auf Graph Größe bei:  -
  - 8bit/16bit hat keinen Einfluss auf Ergebnis bei:     mode=eightbit,mode=weights
  - 8bit/16bit hat keinen Einfluss auf Graph Größe bei:  mode=eightbit,mode=weights

  - 16 bit Flag scheint keinen Einfluss zumindest auf Dateigröße des Graphen zu haben (Bei allen Modi)
  - 16 bit Flag hat Einfluss auf Ergebnis
  - Flag.bitdepth wird nur verwendet in Funktion:
    - quantize_weight_rounded(input_node): (if self.mode == "weights_rounded":)
    - round_nodes_recursively(self, current_node) (if self.mode == "round":) in
    - quantize_weights(self, input_graph, quantization_mode): (if mode=weights or mode=weights_rounded)

#######################################################################################################

convert graph geht nicht:
- how to separate nodes with type DT_FLOAT from other nodes that dont need any casting
- How do I build the graph if I dont change anything and just copy the nodes?
  - just copying and adding to new graph doesnt seem to work...


freeze graph:

output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0
  for input_node in inference_graph.node:
    output_node = node_def_pb2.NodeDef()
    if input_node.name in found_variables:
      output_node.op = "Const"
      output_node.name = input_node.name
      dtype = input_node.attr["dtype"]
      data = found_variables[input_node.name]
      output_node.attr["dtype"].CopyFrom(dtype)
      output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
          tensor=tensor_util.make_tensor_proto(data,
                                               dtype=dtype.type,
                                               shape=data.shape)))
      how_many_converted += 1
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])
  print("Converted %d variables to const ops." % how_many_converted)
  return output_graph_def
