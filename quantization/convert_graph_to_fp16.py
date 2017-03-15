import tensorflow as tf
# import os

from tensorflow.python.platform import gfile

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util

import copy
import numpy as np


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


def run_graph():


  # Create a new TensorFlow computational graph.
  graph = tf.Graph()

  with graph.as_default():
    # Open the graph-def file for binary reading.

    path_fp32 = "/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/tensorflow_inception_graph.pb"
    path_fp16 = "/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph.pb"

    with tf.gfile.FastGFile(path_fp16, 'rb') as file:
      # The graph-def is a saved copy of a TensorFlow graph.
      # First we need to create an empty graph-def.
      graph_def = tf.GraphDef()
      # Then we load the proto-buf file into the graph-def.
      graph_def.ParseFromString(file.read())
      # Finally we import the graph-def to the default TensorFlow graph.
      tf.import_graph_def(graph_def, name='')

      # Now graph holds the Inception model from the proto-buf file.


    y_pred = graph.get_tensor_by_name("softmax:0")

    # Read the jpeg-image as an array of bytes.
    image_path = "panda.jpg"
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Image is passed in as a jpeg-encoded image.
    feed_dict = {"DecodeJpeg/contents:0": image_data}

    session = tf.Session(graph=graph)


    # Execute the TensorFlow session to get the predicted labels.
    options = tf.RunOptions(output_partition_graphs=True)
    metadata = tf.RunMetadata()

    pred = session.run(y_pred, feed_dict=feed_dict, options=options, run_metadata=metadata)

    # Reduce the array to a single dimension.
    np.set_printoptions(threshold=np.inf)
    pred = np.squeeze(pred) # this is the prediction

    # get the indices of the top 5 results
    result = pred.argsort()[-5:][::-1]

    print "Confidence:"
    for i in result:
      print i,": ", pred[i]


    # print "-->partition_graph_def:"
    # for partition_graph_def in metadata.partition_graphs:
    #   for node in partition_graph_def.node:
    #     print node.name  # Contains all the nodes that ran on a single device.



def create_fp16_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network with all float32 Nodes converted to float16 Nodes
  """

  # start session
  with tf.Session() as sess:
    model_filename = "/home/jzuern/tensorflow/tensorflow/examples/label_image/data/tensorflow_inception_graph.pb"
    output_filename = '/home/jzuern/Dropbox/develop/hiwi_mrt/quantization/fp16_graph.pb'

    not_convertable_nodes = ["ResizeBilinear", "Cast", "ExpandDims"]
    input_from_not_convertable_nodes = ["Sub"]
    add_T_fp16_list = ["pool", "pool_1", "mixed_3/pool", "mixed_8/pool", "mixed_10/tower_2/pool"]


    nConvert_nodes = 0




    # import graph file
    print("loading graph...")
    with gfile.FastGFile(model_filename, 'rb') as f:
      input_graph_def = tf.GraphDef()
      input_graph_def.ParseFromString(f.read())

    fp16_graph_def = graph_pb2.GraphDef()

    for old_node in input_graph_def.node:

      new_node = copy.deepcopy(old_node)


      if old_node.name in not_convertable_nodes:
        fp16_graph_def.node.extend([new_node])
        continue


      if old_node.name in input_from_not_convertable_nodes: # found incompatible input node
        print "found input from not convertable node: ", old_node.name
        for input_name in old_node.input: # go through inputs of node
            if input_name in not_convertable_nodes: # found incompatible input
              print "found input: ", input_name
              convert_name = "Convert" + str(nConvert_nodes)
              convert_node = create_node("Cast", convert_name)# Op and Name
              convert_node.input.append(input_name) # add inconvertable node as input for conversion node
              convert_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
              convert_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.half.as_datatype_enum))
              new_node.input.extend([convert_node.name]) # add conversion node as input for node
              new_node.input.remove("Sub/y")
              new_node.input.remove(input_name) # remove original input
              new_node.input.extend(["Sub/y"])
              fp16_graph_def.node.extend([convert_node]) # add inserted conversion node to GraphDef
              nConvert_nodes += 1


      if old_node.attr["dtype"] == attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum): # extract fp32 nodes:
        print "checkpot - found float32 node with dtype", old_node.name
        tensor = old_node.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(tensor)
        tensor_fp16 = tf.cast(tensor_value, tf.half)

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



      # add new fp16 node to new graph
      fp16_graph_def.node.extend([new_node])


    print("%d ops in the final graph." % len(fp16_graph_def.node))


    # write fp16 graph to pb file
    with gfile.FastGFile(output_filename, 'wb') as f:
      f.write(fp16_graph_def.SerializeToString())
    print "writing ", output_filename, " completed."
    return fp16_graph_def


def main(_):

  graph = create_fp16_graph() # create fp16 graph
  run_graph() # run graph


if __name__ == '__main__':
  tf.app.run()
