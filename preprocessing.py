import numpy as np
from sklearn.model_selection import train_test_split
import random as rn

def convert_categorical_to_numbers(filepath):
	conversion = {'\n':0,'':0, '"doctorate degree"':6, '"never smoked"':1, '"tried smoking"':2, '"former smoker"':3, '"current smoker"':4,'"social drinker"':2, '"drink a lot"':3,'"i am often early"':1, '"i am always on time"':2, '"i am often running late"':3,'"never"':1, '"only to avoid hurting someone"':2, '"sometimes"':3, '"everytime it suits me"':4,'"no time at all"':1, '"less than an hour a day"':2, '"few hours a day"':3, '"most of the day"':4,'"female"':1, '"male"':2,'"left handed"':1, '"right handed"':2,'"currently a primary school pupil"':1, '"primary school"':2, '"secondary school"':3, '"college/bachelor degree"':4, '"masters degree"':5, '"no"':1, '"yes"':2,'"city"':1, '"village"':2, '"house/bungalow"\n':1, '"block of flats"\n':2}
	h = [73, 74, 107, 108, 132, 144, 145, 146, 147, 148, 149]

	print('opening file...')
	file = []
	with open(filepath, "r") as f:
	    flag = True
	    for x in f:
	        if flag:
	            flag = False
	            continue
	        file.append(x)

	for line in range(len(file)):
	    file[line] = file[line].split(",")

	print('data stored...')
	print('converting categorical data...')

	with open("formatted_responses.csv", "a") as f1:
	    for i in file:
	        for j in range(len(i)):
	            if j in h:
	                i[j] = conversion[i[j]]
	            elif i[j] == '':
	                i[j] =  rn.randint(1, 5)
	            f1.write(str(i[j]))
	            if not (len(i)-1 == j):
	                f1.write(",")
	        f1.write("\n")

	print('data converted...')
	print('new file "formatted_responses.csv" created')

def create_file(name_of_file, data):

	print('creating ' + name_of_file + '...')
	with open(name_of_file,'a') as f:
	    for i in data:
	        for j in range(len(i)):
	            f.write(str(i[j]))
	            if not (len(i)-1 == j):
	                f.write(",")
	        f.write("\n")
	print('file created...')

def split_files(filepath):

	print('splitting data into train/dev/test files')
	data = np.genfromtxt('formatted_responses.csv', delimiter=",", usecols=np.arange(0,150))
	train,test = train_test_split(data, shuffle=False, test_size=.305)
	test, dev = train_test_split(test, shuffle=False, test_size=.33)

	create_file('responses.tr', train)
	create_file('responses.de', dev)
	create_file('responses.te', test)
	print('data split completed...')

def feature_label_split(file):
	data = np.genfromtxt(file, delimiter=",", usecols=np.arange(0,150))
	Y = data[:,99]
	X = np.delete(data, np.s_[99:100], axis=1)
	return X,Y

def process_data():
	convert_categorical_to_numbers("young-people-survey/responses.csv")
	split_files("formatted_responses.csv")


# smoke_conversion = {'"never smoked"':1, '"tried smoking"':2, '"former smoker"':3, '"current smoker"':4}
# drinking_conversion = {"never":1, "social drinker":2, "drink a lot":3}
# timekeeping_conversion = {"i am often early":1, "i am always on time":2, "i am often running late":3}
# lie_conversion = {"never":1, "only to avoid hurting someone":2, "sometimes":3, "everytime it suits me":4}
# online_conversion = {"no time at all":1, "less than an hour a day":2, "few hours a day":3, "most of the day":4}
# gender_conversion = {"female":1, "male":2}
# mainhand_conversion = {"left handed":1, "right handed":2}
# education_conversion = {"currently a primary school pupil":1, "primary school":2, "secondary school":3, "college/bachelor degree":4, "masters degree":5}
# sibling_conversion = {"no":1, "yes":2}
# childhood_spent_conversion = {"city":1, "village":2}
# childhood_lived_conversion = {'"house/bungalow"\n':1, '"block of flats"\n':2}
