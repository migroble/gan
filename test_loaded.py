import os
import scipy.misc
import numpy as np
import tensorflow as tf

HEIGHT, WIDTH, CHANNEL = 128, 160, 3
INPUT_DIM = 150
TYPE_DIM = 5
BATCH_SIZE = 64
MODEL_DIR = "model/"
SAMPLES_DIR = "images/"

def save_images(images):
	if not os.path.exists(SAMPLES_DIR):
		os.makedirs(SAMPLES_DIR)
	
	trans_images = np.clip(images, 0, 1)
	for image in trans_images:
		scipy.misc.imsave(SAMPLES_DIR + str(len(os.listdir(SAMPLES_DIR))) + ".png", image)

if __name__ == "__main__":
	f = tf.gfile.GFile(MODEL_DIR + "model.pb", "rb")
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name="")
	
	random_input = graph.get_tensor_by_name("input/random_input:0")
	type = graph.get_tensor_by_name("input/type:0")
	is_training = graph.get_tensor_by_name("input/is_training:0")
	
	sess = tf.Session(graph=graph)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)	
	
	test_positions = np.random.uniform(-1, 1, size=[BATCH_SIZE, 150]).astype(np.float32)
	test_types = np.random.uniform(0, 1, size=[BATCH_SIZE, 5]).astype(np.float32)
	images = sess.run("gen/gen:0", feed_dict={random_input: test_positions, type: test_types, is_training: False})
	save_images(images)