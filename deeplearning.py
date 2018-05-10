# coding: utf-8
import tensorflow as tf
import pprint

class CDeepLearningBase(object):
	def __init__(self):
		self.W = tf.Variable(tf.zeros([1, 1]), name="weights")
		self.b = tf.Variable(0.0, name="bias")

	def Inference(self, X):
		"""计算推断模型在X上的输出，根据输入与当前的W与b计算出预测值
		"""
		t = tf.matmul(X, self.W) + self.b
		return tf.reshape(t, [t.shape[0],])

	def Loss(self, X, Y):
		"""根据训练数据，计算并返回当前推断模型的损失
		"""
		Y_Predicted = self.Inference(X)
		loss = tf.squared_difference(Y, Y_Predicted)
		return tf.reduce_sum(loss)

	def Train(self, total_loss):
		learning_rate = 0.00000001
		return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, var_list=[self.W, self.b])

	def Input(self):
		X = []
		Y = []
		return X, Y

	def InitWeight(self, X):
		"""根据输入初始化W参数

		:param X:必须是两维张量
		"""
		shape = [len(X[0]), 1]  # 默认输入是二维张量
		self.W = tf.Variable(tf.zeros(shape), name="weights")

	def Evaluate(self, sess, X, Y):
		print("输出评估值：")
		result = sess.run(self.Inference(X))
		print(result, "\n长度:%s" % len(result))
		print("正确值:%s" % len(Y))
		print(Y)

	def DoLearning(self):
		X, Y = self.Input()
		print("开始训练，训练数据集为：")
		print("特征量：")
		pprint.pprint(X)
		print("标签:")
		pprint.pprint(Y)

		self.InitWeight(X)
		total_loss = self.Loss(X, Y)
		train_op = self.Train(total_loss)

		sess = tf.Session()
		init_all_var = tf.global_variables_initializer()
		sess.run(init_all_var)
		print("初始参数为：W=%s, b=%s" % (sess.run(self.W), sess.run(self.b)))
		# coord = tf.train.Coordinator()    # 线程管理相关，没什么用
		# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		training_steps = 10000
		for step in range(training_steps):
			sess.run([train_op])
			if step % (training_steps/10) == 0:
				print("loss：", sess.run([total_loss]))

		print("训练结束：参数:W=%s, b=%s" % (sess.run(self.W), sess.run(self.b)))
		self.Evaluate(sess, X, Y)
		writer = tf.summary.FileWriter("tensorboard", sess.graph)
		# coord.request_stop()
		# coord.join(threads)
		sess.close()
