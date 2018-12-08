from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from baselines import *
from preprocessing import *
from numpy import *
import pickle

def main():
	models = []
	Xtr, Ytr = feature_label_split("data/responses.tr")

	print()
	print('Training baseline classifier: Most Frequent')
	base = most_frequent()
	base.train(Xtr, Ytr)
	s1 = base.predict_all(Xtr)
	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	print()

	models.append(base)

	print('Training the Random Forest Model')
	print('Max feature = 50')
	print('depth = 5')
	print('random seed = 22')
	print('n estimators = 1000')
	print('using gini impurity')

	rand_forest = RandomForestClassifier(n_estimators=1000,max_features=50, max_depth=5, random_state=22, criterion='gini')
	rand_forest.fit(Xtr,Ytr)
	s1 = rand_forest.predict(Xtr)

	models.append(rand_forest)

	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	print()

	print('Training decision tree classifier')
	print('depth = 5')
	classifier = tree.DecisionTreeClassifier(max_depth=5)
	classifier.fit(Xtr, Ytr)
	s1 = classifier.predict(Xtr)
	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	models.append(classifier)
	print()

	print('Training KNN classifier')
	print('K = 2')
	print('weight = uniform')
	
	# pickle_file = open('rand_forest_classifier.pkl', 'ab')
	# pickle.dump(rand_forest, pickle_file)
	# pickle_file.close()

	
	# print("the accuracy on the development data is :{0}".format(mean(s2==Yde)))
	# print("the accuracy on the testing data is :{0}".format(mean(s3==Yte)))
	# print(mean(s2==Yte))

if __name__ == '__main__':
	main()