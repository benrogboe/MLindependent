#Perceptron algorithm used for classifying a linearly seperable dataset
#using "Iris dataset"
import numpy as np
import numpy.random import seed
import pandas as pd

iris = pd.read_csv('CS/ML(independent)/Iris.csv')
X = iris.data
y = iris.target

test_size = 0.3 #splitting the data
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size = test_size, random_state = random_state
#standardizing the features
sc = StandardScalar
#fits to training data only
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#strings must be transformed to numbers(scikit takes care of this)

class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=10): #will be initialized with every class call niter = no of iterations
        self.eta =eta #float value that signifies learning rate
        self.n_iter = n_iter # no of epochs over training data
    def fit(self,X,y): #will fit trainng data X is feature set and y are target values. X is an nxm matrix with n examples and m features. Y has n samples with a target value asn
        self.w_ = np.zeros(1+X.shape[1]) #will create array(vector) that will hold weights after the model has been fit to the data. Is set to no of columns
        self.errors = [] #keps track of the # of missed classifications for each epoch
        for in range (self.n_iter): #will loop through training data 10x
            errors = 0 #set to 0 for each iteration
            for xi,target in zip(X,y): #will loop through each combination of samples and target for sample will temporarily bind X and y together
                update = self.eta*(target-self.predict(xi)) #we apply learning rate to error froom predictiom
                self.w_[1:] += update* xi #setting weight vector for each feature
                self.w_[0] += update #just apply update to first weight vector index
                errors += int(update !=0.0) #for errors = will add error if after calculating update its not = to 0
            self.errors.append(errors) #append error counter to errors list
        return self
    def net_input(self,X): #takes training features and calculates linear combination of weights and feature vector
        return np.dot(X,self.w_[1:]) + self.w_[0] # ^calculated with the dot product weight vector and initial weight
    def predict(self,X): #returns class labels after net input function
        return np.where(self.net_input(x) >=0.0,1,-1)#if greater than 0.0 will be one, if not, -1...assigns class labels
