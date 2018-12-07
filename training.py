from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
from numpy import *
import pickle

def main():

	Xtr, Ytr = feature_label_split("data/responses.tr")

	print()
	print('running the Random Forest Model')
	print('Max feature = 50')
	print('depth = 5')
	print('random seed = 22')
	print('n estimators = 1000')
	print('using gini impurity')
	print()
	rand_forest = RandomForestClassifier(n_estimators=1000,max_features=50, max_depth=5, random_state=22, criterion='gini')
	rand_forest.fit(Xtr,Ytr)

	print('predicting training data...')
	s1 = rand_forest.predict(Xtr)

	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))

	pickle_file = open('rand_forest_classifier.pkl', 'ab')
	pickle.dump(rand_forest, pickle_file)
	pickle_file.close()
	# print("the accuracy on the development data is :{0}".format(mean(s2==Yde)))
	# print("the accuracy on the testing data is :{0}".format(mean(s3==Yte)))
	# print(mean(s2==Yte))

if __name__ == '__main__':
	main()