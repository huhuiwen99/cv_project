import tensorflow as tf
import numpy as np

class Network(object):
	def __init__(
		self,
		input_dim,
		output_dim,
		learning_rate = 0.001,
	):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.sess = tf.Session()
		self.create_network()
		self.saver = tf.train.Saver(max_to_keep = 0)
		self.model_path = "model/"
		self.sess.run(tf.global_variables_initializer())

	def create_network(self):
		self.input = tf.placeholder(tf.float32, shape = [None, self.input_dim[0], self.input_dim[1], self.input_dim[2]])
		self.label = tf.placeholder(tf.int32, shape = [None])
		self.label_onehot = tf.one_hot(self.label, self.output_dim)
		self.lr = tf.placeholder(tf.float32)

		# (224*224*3)->(223*223*16)
		self.conv1 = tf.layers.conv2d(inputs = self.input, filters = 16, kernel_size = 2, 
			kernel_initializer = tf.truncated_normal_initializer(stddev = 0.001), 
			strides = 1, padding = "same", activation = tf.nn.relu)
		# ->(111*111*16)c
		self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size = 2, strides = 2)
		self.dro1 = tf.nn.dropout(self.pool1, 0.8)
		# ->(110*110*32)
		self.conv2 = tf.layers.conv2d(inputs = self.pool1, filters = 32, kernel_size = 2, 
			kernel_initializer = tf.truncated_normal_initializer(stddev = 0.001),
			strides = 1, padding = "same", activation = tf.nn.relu)
		# ->(55*55*32)
		self.pool2 = tf.layers.max_pooling2d(self.conv2, pool_size = 2, strides = 2)
		self.dro2 = tf.nn.dropout(self.pool2, 0.8)
		# ->(54*54*64)
		self.conv3 = tf.layers.conv2d(inputs = self.pool2, filters = 64, kernel_size = 2, 
			kernel_initializer = tf.truncated_normal_initializer(stddev = 0.001),
			strides = 1, padding = "same", activation = tf.nn.relu)
		# ->(28*28*64)
		self.pool3 = tf.layers.max_pooling2d(self.conv3, pool_size = 2, strides = 2)
		self.dro3 = tf.nn.dropout(self.pool3, 0.8)

		self.flat1 = tf.reshape(self.pool3, [-1, 28*28*64])
		self.flat2 = tf.layers.dense(self.flat1, 216)
		self.output = tf.layers.dense(self.flat2, self.output_dim)

		# print(self.label_onehot.shape)
		# print(self.output.shape)

		self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label_onehot, logits = self.output)
		self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		#这个accuracy操作必须要加上tf.local_variables_initializer()
		self.correct_pred = tf.equal(tf.argmax(self.label_onehot, 1), tf.argmax(self.output, 1))
		# print(self.correct_pred.shape)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


	def save_model(self, episode):
		save_path = self.saver.save(self.sess, (self.model_path + str(episode) + ".ckpt"))

	def restore_model(self, episode):
		oad_path = self.saver.restore(self.sess, (self.model_path + str(episode) + ".ckpt"))








