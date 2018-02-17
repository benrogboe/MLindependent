# k nearest neighbors implementation = compares distance between a particular point and its
# distance from another to determine which class a given datapoint belongs to (beningn or malignanrt)

#Dataset is from UWisconsin breat cancer demographics study

import numpy as np
from sklearn import preprocessing,cross_validation, neighbors
df = pd.read_csv('CS/ML(independent)/breastcancerdata.data.txt') #reading dataset
df.replace('?',-9999, inplace=True)#redefining missing data with a number (-9999 will be treated as an outlier)
df.drop(['id'], 1, inplace=True)#dropping the id column

X = np.array(df.drop(['class'],1))#features(features are all except for class)
y = np.array(df['class']) #labels (label is class)

#cross validation(method by which the training set can be generalized) training data is split into
# n folds and n-1 folds are used to train on, and the remaining fold is used as a "test"
#^but is not from test data. Different combinations of the training data are "checked" with the "test" data and the
#combination with the lowest standard error w/ respect to the test data is used to form a polynomial which will be used
X_train, X_test, y_train, y_test =cross_validation.train_test_split(X,y,test_size=0.2)#cross validation
clf = neighbors.KNeighborsClassifier()#defining the classifier
clf.fit(X_train, y_train) #data is fit
accuracy = clf.score(X_test, y_test)#can test its performance
print(accuracy)
example_measures = np.array([2,3,1,1,1,6,3,2,6])#example to make prediction about

prediction = clf.predict(example_measures)#classifying example
print (prediction) 
