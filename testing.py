import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
from numpy import *

def main():
	Xte, Yte = feature_label_split("data/responses.te")
	print('Getting test data...')
	print()
	pickle_file = open('models.pkl', 'rb')
	classifiers = pickle.load(pickle_file)

	print('predicting test sample data with mostfrequent.')
	mostfrequent = classifiers[0]
	s1 = mostfrequent.predict(Xte)
	print("the accuracy on the testing data is :{0}".format(mean(s1==Yte)))
	print()
	print('predicting test sample data with forest.')
	forest = classifiers[1]
	s2 = forest.predict(Xte)
	print("the accuracy on the testing data is :{0}".format(mean(s2==Yte)))
	print()
	print('predicting test sample data with decision tree.')
	tree = classifiers[2]
	s3 = tree.predict(Xte)
	print("the accuracy on the testing data is :{0}".format(mean(s3==Yte)))
	print()
	print('predicting test sample data with knn.')
	knn = classifiers[3]
	s4 = knn.predict(Xte)
	print("the accuracy on the testing data is :{0}".format(mean(s4==Yte)))
	print()
	print('predicting test sample data with svc.')
	svc = classifiers[4]
	s5 = svc.predict(Xte)
	print("the accuracy on the testing data is :{0}".format(mean(s5==Yte)))
	print()
	pickle_file.close()

if __name__ == '__main__':
	main()
