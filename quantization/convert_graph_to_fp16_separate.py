import tensorflow as tf
# import os
# import numpy as np

from tensorflow.python.platform import gfile

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util

import copy


def create_node(op, name, inputs=None):

  new_node = node_def_pb2.NodeDef()
  new_node.op = op
  new_node.name = name
  if inputs:
    for input_name in inputs:
      new_node.input.extend([input_name])
  return new_node

def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape)))
  except KeyError:
    pass

def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass

def create_fp16_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network with all float32 tensors converted to float16 tensors
  """

  # start session
  with tf.Session() as sess:
    model_filename = "/home/jzuern/tensorflow/tensorflow/examples/label_image/data/tensorflow_inception_graph.pb"
    output_filename = '/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph_separate.pb'


    not_convertable_nodes = ["Mul"]
    input_from_not_convertable_nodes = ["conv/Conv2D"]

    add_T_fp16_list = ["pool", "pool_1", "mixed_3/pool", "mixed_8/pool", "mixed_10/tower_2/pool"]
    nConvert_nodes = 0

    first_nodes_list = ["DecodeJpeg/contents", "DecodeJpeg", "Cast", "ExpandDims/dim", "ExpandDims",
                   "ResizeBilinear/size", "ResizeBilinear",  "Sub", "Sub/y", "Mul", "Mul/y"]


    # import graph file
    print("loading graph...")
    with gfile.FastGFile(model_filename, 'rb') as f:
      input_graph_def = tf.GraphDef()
      input_graph_def.ParseFromString(f.read())

    fp16_graph_def = graph_pb2.GraphDef()

    print "all attributes, ... of GraphDef: ", dir(fp16_graph_def)

    for old_node in input_graph_def.node:

      new_node = node_def_pb2.NodeDef()
      new_node.CopyFrom(old_node)

      if  new_node.name == "Mul" or new_node.name == "conv/Conv2D" or new_node.name == "conv/conv2d_params":
        print "checking node ", new_node.name
        print new_node





      if new_node.name in first_nodes_list: # skip first couple of nodes
        fp16_graph_def.node.extend([old_node])
        print "skipping Node ", old_node.name
        continue


      if old_node.name in input_from_not_convertable_nodes: # found incompatible input node
        print "found input from not convertable node in node: ", old_node.name
        for input_name in old_node.input: # go through inputs of node
            if input_name in not_convertable_nodes: # found incompatible input
              print "found input: ", input_name
              convert_name = "Convert" + str(nConvert_nodes)
              convert_node = create_node("Cast", convert_name)# Op and Name
              convert_node.input.append(input_name) # add inconvertable node as input for conversion node
              convert_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
              convert_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))
              new_node.input.remove(input_name) # remove original input
              new_node.input.remove("conv/conv2d_params")# remove other input

              new_node.input.extend([convert_node.name]) # add conversion node as input for node
              new_node.input.extend(["conv/conv2d_params"])# add other input again
              fp16_graph_def.node.extend([convert_node]) # add inserted conversion node to GraphDef
              nConvert_nodes += 1


      if old_node.attr["dtype"] == attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum): # extract fp32 nodes:
        print "checkpot - found float32 node with dtype", old_node.name
        tensor = old_node.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(tensor)
        tensor_fp16 = tf.cast(tensor_value, tf.half)
        #tensor_fp16 = tf.to_bfloat16(tensor_value) # alternative way of casting to float16

        new_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
          tensor_fp16.eval(),
          dtype=dtypes.half,
          shape=tensor_fp16.shape)))

        new_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))


      if old_node.name in add_T_fp16_list:
        new_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))


      if old_node.attr["T"] == attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum): # extract fp32 nodes:
        print "checkpot - found float32 node with T", old_node.name
        new_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))
        # print old_node

      if old_node.attr["DstT"] == attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum): # extract fp32 nodes:
        print "checkpot - found float32 node with DstT", old_node.name
        new_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))


      # BACKUP: JUST COPY NODE
    #   new_node = node_def_pb2.NodeDef()
    #   new_node.CopyFrom(old_node)

      if  new_node.name == "Mul" or new_node.name == "conv/Conv2D" or new_node.name == "conv/conv2d_params":
        print "after conversion: checking node ", new_node.name
        print new_node


      # add new fp16 node to new graph
      fp16_graph_def.node.extend([new_node])

    # copy old graph into new graph:
    # fp16_graph_def = input_graph_def # dummy


    print("%d ops in the final graph." % len(fp16_graph_def.node))


    # write fp16 graph to pb file
    with gfile.FastGFile(output_filename, 'wb') as f:
      f.write(fp16_graph_def.SerializeToString())


def main(_):

  create_fp16_graph()


if __name__ == '__main__':
  tf.app.run()




# ### in FREEZE_GRAPH.PY
#
# output_graph_def = graph_pb2.GraphDef()
#   for input_node in inference_graph.node:
#     output_node = node_def_pb2.NodeDef()
#     output_node.op = "Const"
#     output_node.name = input_node.name
#     dtype = input_node.attr["dtype"]
#     data = found_variables[input_node.name]
#     output_node.attr["dtype"].CopyFrom(dtype)
#     output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
#           tensor=tensor_util.make_tensor_proto(data,
#                                                dtype=dtype.type,
#                                                shape=data.shape)))
#     output_graph_def.node.extend([output_node])
#   return output_graph_def
