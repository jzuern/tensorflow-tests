  def rewrite(self, output_node_names):
    """Triggers rewriting of the float graph.
    Args:
      output_node_names: A list of names of the nodes that produce the final
        results.
    Returns:
      A quantized version of the float graph.
    """
    self.output_graph = graph_pb2.GraphDef()
    output_nodes = [
        self.nodes_map[output_node_name]
        for output_node_name in output_node_names
    ]
    if self.mode == "round":
      self.already_visited = {}
      for output_node in output_nodes:
        self.round_nodes_recursively(output_node)





    elif self.mode == "quantize":
      self.already_visited = {}
      self.already_quantized = {}
      for output_node in output_nodes:
        self.quantize_nodes_recursively(output_node)






    elif self.mode == "eightbit":
      self.set_input_graph(graph_util.remove_training_nodes(self.input_graph))
      output_nodes = [
          self.nodes_map[output_node_name]
          for output_node_name in output_node_names
      ]

      self.state = EightbitizeRecursionState(
          already_visited={}, output_node_stack=[], merged_with_fake_quant={})
      for output_node in output_nodes:
        self.eightbitize_nodes_recursively(output_node)
      self.state = None
      if self.input_range:
        self.add_output_graph_node(
            create_constant_node("quantized_input_min_value", self.input_range[
                0], dtypes.float32, []))
        self.add_output_graph_node(
            create_constant_node("quantized_input_max_value", self.input_range[
                1], dtypes.float32, []))
      if self.fallback_quantization_range:
        self.add_output_graph_node(
            create_constant_node("fallback_quantization_min_value",
                                 self.fallback_quantization_range[0],
                                 dtypes.float32, []))
        self.add_output_graph_node(
            create_constant_node("fallback_quantization_max_value",
                                 self.fallback_quantization_range[1],
                                 dtypes.float32, []))
      if FLAGS.strip_redundant_quantization:
        self.output_graph = self.remove_redundant_quantization(
            self.output_graph)
        self.remove_dead_nodes(output_node_names)
      self.apply_final_node_renames()






    elif self.mode == "weights":
      self.output_graph = self.quantize_weights(self.input_graph,
                                                b"MIN_COMBINED")
      self.remove_dead_nodes(output_node_names)





    elif self.mode == "weights_rounded":
      self.output_graph = self.quantize_weights(self.input_graph, self.mode)
      self.remove_dead_nodes(output_node_names)




    else:
      print("Bad mode - " + self.mode + ".")
    return self.output_graph
