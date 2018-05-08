# coding: utf-8
import tensorflow as tf
import deeplearning


def Main():
	test_deep_learning = TestDeepLearning()
	test_deep_learning.DoLearning()


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


if __name__ == "__main__":
	Main()



