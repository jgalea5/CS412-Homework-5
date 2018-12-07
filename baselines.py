from binary_class import *
import numpy as np

class most_frequent(BinaryClassifier):

	def __init__(self):
		# assume class 1, which is "very empathetic"
		self.most_frequent_class = 1

	def predict(self, X):
		return self.most_frequent_class

	def predict_all(self, X):
		# print(X.shape)
		d1, d2 = X.shape
		result = np.zeros(d1)
		for i in range(d1):
			result[i] = self.most_frequent_class
		return result

	def train(self,X,Y):
		#label = {0:"not very empathetic", 1:"not very empathetic", 2:"not very empathetic", 3:"not very empathetic", 4:"very empathetic", 5:"very empathetic"}
		empathetic_count = 0
		not_empathetic_count = 0
		for y in Y:
			if y == -1:#label[y] == "not very empathetic":
				not_empathetic_count += 1
			elif y == 1:#label[y] == "very empathetic":
				empathetic_count += 1
		# print(empathetic_count)
		# print(not_empathetic_count)

		if empathetic_count > not_empathetic_count:
			self.most_frequent_class = 1
		else:
			self.most_frequent_class = 0