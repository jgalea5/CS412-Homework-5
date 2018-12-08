from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
	dtclassifier = tree.DecisionTreeClassifier(max_depth=5)
	dtclassifier.fit(Xtr, Ytr)
	s1 = dtclassifier.predict(Xtr)
	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	models.append(dtclassifier)
	print()

	print('Training KNN classifier')
	print('K = 2')
	print('weight = uniform')
	knnclf = KNeighborsClassifier(n_neighbors=2, weights='uniform')
	knnclf.fit(Xtr, Ytr)
	s1 = knnclf.predict(Xtr)
	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	models.append(knnclf)
	print()

	print('Training SVC classifier')
	print('C = 100')
	print('Kernel is poly')
	print('degree = 2')
	svcclassifier = SVC(C = 100, kernel='poly', degree=2)
	svcclassifier.fit(Xtr, Ytr)
	s1= svcclassifier.predict(Xtr)
	print("the accuracy on the training data is :{0}".format(mean(s1==Ytr)))
	models.append(svcclassifier)
	print()

	print("Saving the modesl to a pkl file...")
	pickle_file = open('models.pkl', 'ab')
	pickle.dump(models, pickle_file)
	pickle_file.close()
	print("Closing pkl file...")

if __name__ == '__main__':
	main()
