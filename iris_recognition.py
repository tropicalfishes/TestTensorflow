# coding: utf-8
import tensorflow as tf
import numpy as np
import iris_data

FEATURE_KEYS = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
LABEL_KEY = ['Setosa', 'Versicolor', 'Virginica']

def main():
	train_steps = 1000

	# Feature columns describe how to use the input.
	feature_columns = []
	for key in FEATURE_KEYS:
		feature_columns.append(tf.feature_column.numeric_column(key=key))

	# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=feature_columns,
		# Two hidden layers of 10 nodes each.
		hidden_units=[10, 10],
		# The model must choose between 3 classes.
		n_classes=len(LABEL_KEY),
		model_dir="iris_recognition_model",
	)

	train_feature, train_label = Input()
	# Train the Model.
	classifier.train(
		input_fn=Input,
		steps=train_steps)

	# Evaluate the Model.
	evaluate_feature, evaluate_label = input_evaluation_set()
	print("看看评估的输入：", evaluate_feature)
	print("评估的label", evaluate_label)
	evaluate_result = classifier.evaluate(input_fn=input_evaluation_set, steps=10)
	print("评估结束，结果：", evaluate_result)
	
	# 预测
	features = {'SepalLength': np.array([6.4, 5.0]),
	            'SepalWidth': np.array([2.8, 2.3]),
	            'PetalLength': np.array([5.6, 3.3]),
	            'PetalWidth': np.array([2.2, 1.0])}
	labels = np.array([2, 1])
	predict_result = classifier.predict(
		input_fn=lambda : features,
	)
	for i in range(4):
		print("预测结果：", predict_result.__next__())

def input_evaluation_set():
	features = {'SepalLength': np.array([6.4, 5.0]),
				'SepalWidth':  np.array([2.8, 2.3]),
				'PetalLength': np.array([5.6, 3.3]),
				'PetalWidth':  np.array([2.2, 1.0])}
	labels = np.array([2, 1])
	return features, labels

def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.

	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	return dataset.shuffle(1000).repeat().batch(batch_size)

def evaluate_input_fn(features, labels):
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	return dataset


def Input():
	import pandas as pd
	train_path = "iris_training.csv"
	train = pd.read_csv(train_path, names=FEATURE_KEYS+["Species",], header=0)
	train_y = train.pop("Species")
	train_x = train
	print("看看训练X", train_x)
	print("看看训练Y", train_y)
	d = {}
	for key in FEATURE_KEYS:
		d[key] = train_x[key].values
	print("看看转格式后", d)

	return d, train_y