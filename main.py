# coding: utf-8
import tensorflow as tf
import deeplearning


def Main():
	# test_deep_learning = TestDeepLearning()
	# test_deep_learning.DoLearning()

	# TestTFApi()

	import iris_recognition
	iris_recognition.main()


def TransData():
	path = "fatdata.txt"
	f = open(path, "r")
	s = f.read()
	f.close()
	lines = s.split("\n")
	fatdata = {}
	fs = "DATA={\n"
	for line in lines:
		lData = line.split()
		fatdata[int(lData[0])] = {"weight":int(lData[2]), "age":int(lData[3]), "fat":int(lData[4])}
		fs += "%s:{'weight':%s, 'age':%s, 'fat':%s},\n" % (lData[0], lData[2], lData[3], lData[4])
	fs += "}"
	f = open("fatdata.py", "w")
	f.write(fs)
	f.close()


class TestDeepLearning(deeplearning.CDeepLearningBase):
	def Input(self):
		import fatdata
		weight_age = []
		fat = []
		for data in fatdata.DATA.values():
			weight_age.append([float(data["weight"]), float(data["age"])])
			fat.append(float(data["fat"]))
		return weight_age, fat

def TestTFApi():
	import numpy as np
	feature = {"a":[1.0,2.0,3.], "b":[1.1, 2.2, 3.3]}
	labels = np.array([1.3, 2.3, 3.3])
	dataset = tf.data.Dataset.from_tensor_slices((feature,labels)).shuffle(1000).repeat(10)
	iterator = dataset.make_one_shot_iterator()
	e = iterator.get_next()
	s = tf.Session()
	for i in range(30):
		print("AAA", s.run(e))
	# print("BBB", s.run(e))
	# print("CCC", s.run(e))
	# print("DDD", s.run(e))

if __name__ == "__main__":
	Main()



