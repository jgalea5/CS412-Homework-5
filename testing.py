import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
from numpy import *

def main():
	Xte, Yte = feature_label_split("data/responses.te")
	print('Getting test data...')
	print('retrieving trained random forest model...')
	pickle_file = open('rand_forest_classifier.pkl', 'rb')
	classifier = pickle.load(pickle_file)
	s1 = classifier.predict(Xte)
	pickle_file.close()
	print('predicting test sample data...')

	print("the accuracy on the testing data is :{0}".format(mean(s1==Yte)))

if __name__ == '__main__':
	main()