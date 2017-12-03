## Note that this seems to work as expected when all
## predictors continuous. Should be able to change metric
## and get reasonable results for all categorical.
## How it behaves on mix of continuous and categorical
## predictors is to be seen but it seems clear it will be
## heavily dependent on the amount of variation present
## in the continuous 

import numpy as np

class knn_classifier():

    def __init__(self, k):

        # make the object
        # should I be considering more than this?
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, y, X):

        # should basically just be loading?
        # there really isn't anything going on at this stage
        self.X_train = X
        self.y_train = y


    def predict(self, X):

        # generate empty array to hold predictions dim: n x 1 array
        prediction = np.zeros(X.shape[0])

        # loop over all observations (rows) in test set
        for i in range(X.shape[0]):

            # row-wise computation of vector norm of difference between
            # current obs. and all training obs. will be an n x 1 array
            distance = np.linalg.norm(self.X_train - X[i,], axis = 1)

            # keep the labels of closest k obs. from training set
            close_k = self.y_train[np.argsort(distance)][:self.k]

            # find the mode of closest obs. labels
            values, counts = np.unique(close_k, return_counts=True)

            # insert our prediction for ith obs. into ith position of prediciton
            prediction[i] = values[np.argmax(counts)]

        return prediction
