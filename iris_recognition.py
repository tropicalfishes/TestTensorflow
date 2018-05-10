# coding: utf-8
import tensorflow as tf
import numpy as np
import iris_data

def main():
	train_x, labels = input_evaluation_set()
	
	# Feature columns describe how to use the input.
	my_feature_columns = []
	for key in train_x.keys():
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))
	# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each.
		hidden_units=[10, 10],
		# The model must choose between 3 classes.
		n_classes=3)
	
	# Train the Model.
	classifier.train(
		input_fn=lambda: iris_data.train_input_fn(train_x, train_y, args.batch_size),
		steps=args.train_steps)

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

