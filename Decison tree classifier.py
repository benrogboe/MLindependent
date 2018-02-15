'''Decision tree classifier using scikit-learn tools'''
#Using a spotify dataset which provides song attributes
#more specifically dataset is a list of songs with target value of  0 or 1(like
#or dislike)-want to predict wether user will like song from test set based on
#training from training set features consist of duration of song, dancability,
#loudness, so some are discrete and some are continuous
import pandas as pd #presents data in tabular format(if I want to analyze data set at some point)
import numpy as np #matrix manipulation
from sklearn import true
from sklearn.tree import DecisionTreeClassifier #tools and functionality for decision trees
from sklearn.model_selection import train_test_split #splitting training data from test data

#Importing data from file on desktop that will be used to train the classifier
data = pd.read_csv('CS/ML(independent)/data.csv')
train, test = train_test_split(data, test_size = 0.15)# Randomly splitting the training and testing data and assigning variables

d = DecisionTreeClassifier(min_samples_split=100) #100 denotes the number of samples needed to split a node
feat = ['loudness', 'danceability','valence','energy','instrumentalness','acousticness','key','speechiness','duration_ms']
#^features that should be a factor in the formation of decision tree
x_train = train[feat] # features to train on (scikit syntax)
y_train = train['target'] # labels(for training)

x_test = test[feat] # features for teeting on
y_test= test['target'] # labels(for testing)

dt = d.fit(x_train, y_train) #constructing the decision tree, determining feature to start with/ split on is not manually coded
#now one should be able to pass in parameters(x) and recieve a prediction
y_pred = d.predict(x_test) #will construct an array of target values(0,1) that denote wether a goven song from the test
#data is liked or not
