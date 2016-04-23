import numpy as np 
import pandas as pd
from numpy import genfromtxt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 


# my_data = pd.read_csv('testset.txt')

# # my_data.loc[my_data["class"] == "Iris-setosa", "class"] = 0
# # my_data.loc[my_data["class"] == "Iris-versicolor", "class"] = 1
# # my_data.loc[my_data["class"] == "Iris-virginica", "class"] = 2

# print my_data["sl"]

iris = load_iris()
regression = LogisticRegression()

iris_data = iris.data
iris_target = iris.target

index = []
#seperating the training and testing
for i in range (0,25):
	index.append(i)
for i in range (51,75):
	index.append(i)
for i in range (101,125):
	index.append(i)

#training data
train_target = np.delete(iris.target, index)
train_data = np.delete(iris.data, index, axis =0)

#testing data 
test_target = iris.target[index]
test_data = iris.data[index]


#finding the fit for the data
fit = regression.fit(train_data, train_target)

#print test data
print iris.feature_names
# print iris.target_names
print train_data

#testing the data according to the fit
test = regression.predict(test_data)

#checking the percentage of match
result = test == test_target

print "result =" , result

y = 0
z = 0
percent = 0.00

for i in result:
	if i == True: 
		y = y + 1
	else:
		z = z + 1

print "correct prediction", y
print "incorrect prefiction", z

percent = (y*1.000/(z+y))*100 			

print percent , "%"
