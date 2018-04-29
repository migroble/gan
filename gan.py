import os
import json
import time
import scipy.misc
import numpy as np
import tensorflow as tf

HEIGHT, WIDTH, CHANNEL = 128, 160, 3
BATCH_SIZE = 64
EPOCH = 5000
INPUT_DIM = 150
SAVE_DIR = "model"
LOG_DIR = "log"
SAMPLES_DIR = "samples"
DATASET_DIR = "dataset"
d_iterations = 5
g_iterations = 2
types = {
	"blackholes":	[1, 0, 0, 0, 0],
	"galaxies":		[0, 1, 0, 0, 0],
	"nebulae":		[0, 0, 1, 0, 0],
	"starclusters":	[0, 0, 0, 1, 0],
	"stars":		[0, 0, 0, 0, 1]
}
type_arr = [key for key in types]
TYPE_DIM = len([key for key in types])
test_positions = json.load(open("pos.txt"))["positions"]
test_types = json.load(open("types.txt"))["types"]

def process_data():
	cwd = os.getcwd()
	dataset_dir = os.path.join(cwd, DATASET_DIR)
	
	images = []
	labels = []
	for f in os.listdir(dataset_dir):
		images.append(os.path.join(dataset_dir, f))
		
		type = os.path.splitext(f)[0].split(".")[0]
		labels.append(types[type])
	
	all_images = tf.convert_to_tensor(images, dtype=tf.string)
	all_labels = tf.convert_to_tensor(labels, dtype=tf.float32)
	queue = tf.train.slice_input_producer([all_images, all_labels])
	
	content = tf.read_file(queue[0])
	image = tf.image.decode_jpeg(content, channels=CHANNEL)
	image = tf.image.resize_images(image, [HEIGHT, WIDTH])
	image.set_shape([HEIGHT, WIDTH, CHANNEL])
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta=0.1)
	image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
	image = tf.cast(image, tf.float32)
	image = image / 255.0
	image = tf.clip_by_value(image, 0, 1)
	
	image_batch, label_batch = tf.train.shuffle_batch([image, queue[1]], batch_size=BATCH_SIZE, num_threads=4, capacity=200 + 3 * BATCH_SIZE, min_after_dequeue=200)
	num_images = len(images)
	
	return image_batch, label_batch, num_images

def save_images(images, epoch):
	if not os.path.exists(SAMPLES_DIR):
		os.makedirs(SAMPLES_DIR)
	if not os.path.exists(SAMPLES_DIR + "/epoch " + str(epoch)):
		os.makedirs(SAMPLES_DIR + "/epoch " + str(epoch))
	
	trans_images = np.clip(images, 0, 1)
	for i, image in enumerate(trans_images):
		scipy.misc.imsave(SAMPLES_DIR + "/epoch " + str(epoch) + "/" + str(i) + ".png", image)

def trim(x, decimals=2):
	return str(int(x * 10**decimals) / 10**decimals)

def elapsed_time(before, now):
	delta = now - before
	if delta > 60:
		elapsed = str(int(delta / 60)) + " min " + str(int(delta%60)) + " s"
	else:
		elapsed = trim(delta) + " s"
	
	return elapsed

def gen(input, type, is_training, reuse=False):
	with tf.variable_scope("gen") as scope:
		if reuse:
			scope.reuse_variables()
		
		h, w = 4, 5
		c1, c2, c3, c4, c5 = 160, 80, 40, 20, 10
		
		cond = tf.concat([input, type], 1, name="cond")
		
		w1 = tf.get_variable("w_g", shape=[INPUT_DIM + TYPE_DIM, h * w * c1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		b1 = tf.get_variable("b_g", shape=[h * w * c1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		flat_conv1 = tf.add(tf.matmul(cond, w1), b1, name="flat_conv1")
		
		conv1 = tf.reshape(flat_conv1, shape=[-1, h, w, c1], name="conv1")
		bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn1")
		act1 = tf.nn.relu(bn1, name="act1")
		
		conv2 = tf.layers.conv2d_transpose(act1, c2, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv2")
		bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn2")
		act2 = tf.nn.relu(bn2, name="act2")
		
		conv3 = tf.layers.conv2d_transpose(act2, c3, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv3")
		bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn3")
		act3 = tf.nn.relu(bn3, name="act3")
		
		conv4 = tf.layers.conv2d_transpose(act3, c4, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv4")
		bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn4")
		act4 = tf.nn.relu(bn4, name="act4")
		
		conv5 = tf.layers.conv2d_transpose(act4, c5, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv5")
		bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn5")
		act5 = tf.nn.relu(bn5, name="act5")
		
		conv6 = tf.layers.conv2d_transpose(act5, CHANNEL, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),  name="conv6")
		act6 = tf.nn.tanh(conv6, name="gen")
		
		return act6

def dis(input, type, is_training=False, reuse=False):
	with tf.variable_scope("dis") as scope:
		if reuse:
			scope.reuse_variables()
		
		c4, c3, c2, c1 = 160, 80, 40, 20
		
		conv1 = tf.layers.conv2d(input, c1, kernel_size=5, strides=4, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name="conv1")
		act1 = tf.nn.leaky_relu(conv1, name="act1")
		
		conv2 = tf.layers.conv2d(act1, c2, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv2")
		bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn2")
		act2 = tf.nn.leaky_relu(bn2, name="act2")
		
		conv3 = tf.layers.conv2d(act2, c3, kernel_size=5, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv3")
		bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn3")
		act3 = tf.nn.leaky_relu(bn3, name="act3")
		
		conv4 = tf.layers.conv2d(act3, c4, kernel_size=5, strides=1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv4")
		bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training, epsilon=1e-5, decay=0.9,  updates_collections=None, scope="bn4")
		act4 = tf.nn.leaky_relu(bn4, name="act4")
		
		dim = int(np.prod(act4.get_shape()[1:]))
		flat = tf.reshape(act4, shape=[-1, dim], name="flat")
		
		cond = tf.concat([flat, type], 1, name="cond")
		w1 = tf.get_variable("w_d", shape=[cond.shape[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		b1 = tf.get_variable("b_d", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		
		logits = tf.add(tf.matmul(cond, w1), b1, name="dis")
		
		return logits

def train():
	# setting up loss functions and trainers
	with tf.variable_scope("input") as scope:
		real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name="real_image")
		random_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name="random_input")
		type = tf.placeholder(tf.float32, shape=[None, TYPE_DIM], name="type")
		is_training = tf.placeholder(tf.bool, name="is_training")
	#global_step = tf.Variable(0, name="global_step", trainable=False)
	
	fake_image = gen(random_input, type, is_training=is_training)
	
	real_result = dis(real_image, type, is_training=is_training)
	fake_result = dis(fake_image, type, is_training=is_training, reuse=True)
	
	d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
	g_loss = -tf.reduce_mean(fake_result)
	
	#d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_result, labels=tf.ones_like(real_result)))
	#d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result, labels=tf.zeros_like(fake_result)))
	#d_loss = d_loss1 + d_loss2
	
	#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_result, labels=tf.ones_like(fake_result)))
	
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if "dis" in var.name]
	g_vars = [var for var in t_vars if "gen" in var.name]
	
	#trainer_d = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(d_loss, var_list=d_vars)
	#trainer_g = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(g_loss, var_list=g_vars)
	
	trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
	trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
	
	d_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]
	
	# batches
	image_batch, label_batch, samples_num = process_data()
	num_batches = samples_num // BATCH_SIZE
	
	# setting up session
	sess = tf.Session()
	saver = tf.train.Saver(max_to_keep=None)
	
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	
	ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
	path = ckpt.model_checkpoint_path
	if ckpt and path:
		ckpt_name = os.path.basename(path)
		saver.restore(sess, path)
	
	#starting_epoch = sess.run(global_step)
	starting_epoch = int(path.split("-")[1])
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
	
	# training
	before = time.time()
	print(samples_num, "training samples for", EPOCH, "epoch with batch size", BATCH_SIZE, "(", num_batches, "batches ) starting at epoch", starting_epoch)
	try:
		for i in range(starting_epoch, EPOCH):
			before_epoch = time.time()	
			for j in range(num_batches):
				print("epoch", i, ": batch", j, "/", num_batches)
				before_batch = time.time()
				train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, INPUT_DIM]).astype(np.float32)
				
				for k in range(d_iterations):
					train_image, train_label = sess.run([image_batch, label_batch])
					
					sess.run(d_clip)
					
					_, dLoss = sess.run([trainer_d, d_loss], feed_dict={random_input: train_noise, type: train_label, real_image: train_image, is_training: True})
				
				for k in range(g_iterations):
					train_label = sess.run(label_batch)
					
					_, gLoss = sess.run([trainer_g, g_loss], feed_dict={random_input: train_noise, type: train_label, is_training: True})
				
				now_batch = time.time()
				print("\t\ttime elapsed:", elapsed_time(before_batch, now_batch))
			
			now_epoch = time.time()
			print("\ttime elapsed:", elapsed_time(before_epoch, now_epoch), "\n")
			
			if i%10 == 0:
				if not os.path.exists(SAVE_DIR):
					os.makedirs(SAVE_DIR)
				
				saver.save(sess, "./" + SAVE_DIR + "/gan.ckpt", global_step=i)
				
				images = sess.run(fake_image, feed_dict={random_input: test_positions, type: test_types, is_training: False})
				save_images(images, i)
			
			#sess.run(tf.assign_add(global_step, 1))
	except KeyboardInterrupt:
		print("\ntraining ended due to a manual interrupt")
		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)
		
		saver.save(sess, "./" + SAVE_DIR + "/gan.ckpt", global_step=i)
		
		images = sess.run(fake_image, feed_dict={random_input: test_positions, type: test_types, is_training: False})
		save_images(images, i)
		
		now = time.time()
		print("total training time:", elapsed_time(before, now))
	else:
		print("done training")
	
	coord.request_stop()
	coord.join(threads)

def test():
	pass

if __name__ == "__main__":
	train()