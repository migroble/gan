import os
import scipy.misc
import numpy as np
import tensorflow as tf

MODEL_DIR = "model/"

if __name__ == "__main__":
	ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
	path = ckpt.model_checkpoint_path
	
	with tf.Session(graph=tf.Graph()) as sess:
		saver = tf.train.import_meta_graph(path + ".meta", clear_devices=True)
		saver.restore(sess, path)
		default_graph = tf.get_default_graph().as_graph_def()
		#vars = [n.name for n in default_graph.node if "gen/" in n.name or "input/" in n.name]
		graph = tf.graph_util.extract_sub_graph(default_graph, ["gen/gen"])
		for node in graph.node:			
			if node.op == "RefSwitch":
				node.op = "Switch"
				for index in range(len(node.input)):
					if "moving_" in node.input[index]:
						node.input[index] = node.input[index] + "/read"
			elif node.op == "AssignSub":
				node.op = "Sub"
				if "use_locking" in node.attr: del node.attr["use_locking"]
			elif node.op == "AssignAdd":
				node.op = "Add"
				if "use_locking" in node.attr: del node.attr["use_locking"]
		
		print([n.name for n in graph.node])
		output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, ["gen/gen"])
		
		with tf.gfile.GFile(MODEL_DIR + "model.pb", "wb") as f:
			f.write(output_graph_def.SerializeToString())