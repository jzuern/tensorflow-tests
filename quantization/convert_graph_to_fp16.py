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
    output_filename = '/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph.pb'

    # import graph file
    print("loading graph...")
    with gfile.FastGFile(model_filename, 'rb') as f:
      input_graph_def = tf.GraphDef()
      input_graph_def.ParseFromString(f.read())

    fp16_graph_def = graph_pb2.GraphDef()

    for old_node in input_graph_def.node:
      print "..."

      c = False
      if c == True:
        print "hello"
    #   if old_node.attr["dtype"].type == 1 and convert == True: # fp32 node
    #     print "NODE IS A FLOAT NODE AND NOT IN DO-Not-CONVERT-LIST - CONVERT IT!"
    #     # if convert == True:
    #     #     print old_node.name
    #     #     print old_node.attr["dtype"].type
    #
    #     new_node = create_node(old_node.op, old_node.name, [])
    #     # set_attr_dtype(new_node, "dtype", dtypes.float32) # TODO: float32 to float16
    #
    #     # make ndarray from tensor
    #     tensor = old_node.attr["value"].tensor
    #
    #     # convert tensor fp32 to tensor fp16
    #     # tensor = tf.cast(tensor, tf.dtypes.float16)
    #
    #     tensor_value = tensor_util.MakeNdarray(tensor)
    #
    #     # make TensorShape
    #     shape = tensor.tensor_shape
    #     tensor_shape_list = tensor_util.TensorShapeProtoToList(shape)
    #
    #     set_attr_tensor(node=new_node, key='value', value=tensor_value, dtype=dtypes.float32, shape=tensor_shape_list) # TODO: float32 to float16

      else:
        print "OTHERISE, DO NOTHING!" # fp16 node
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(old_node)
    #   # add new fp16 node to new graph

      fp16_graph_def.node.extend([old_node])


    # copy old graph into new graph:
    fp16_graph_def = input_graph_def # dummy


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


        # copy node attribute
        # new_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float16.as_datatype_enum))
        # new_node_tensor_fp16 = tf.cast(new_node_tensor, tf.float16)
        # new_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(new_node_tensor_fp16)))
        # new_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([1, 2], dtypes.float16, [1, 2,]))) # ONLY DUMMY!!




##########################################################################################################

#     print node.attr['dtype']
#     if node.attr['dtype'] == "float32":
#         print "found float32 tensor"
#         tensor = tf.convert_to_tensor(node.attr['value'].tensor.tensor_content)
#         tensor = tf.decode_raw(tensor, tf.float32)
#         print "32 bit tensor values:"
#         print tensor.eval()
#
#         tensor_fp16 = tf.cast(tensor, tf.float16)
#         print "16 bit tensor values:"
#         print tensor_fp16.eval()


## OLE HINWEIS:

# output_graph = graph_pb2.GraphDef()
#
# for orig_node in node_map:
#   new_node = node_def_pb2.NodeDef()
#   new_node.op = orig_node.op
#   new_node.name = orig_node.name
#   new_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float16.as_datatype_enum))
#   for input_name in orig_node.inputs:
#     new_node.input.extend([input_name])
#   output_graph.node.extend([new_node])
