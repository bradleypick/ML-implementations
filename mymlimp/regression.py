#! usr/bin/env python

import numpy as np

class linear_model():

    def __init__(self):

        self.w = None

    def fit(self, X, y, reg=None, lam=0.001):
        """
        External method for fitting linear model
        Inputs:
         - X:   feature (design) matrix of dimension n x d
         - y:   reponse variable (array of length n)
         - reg: type of regularization to use (default is None)
        Output:
         - None, called for side effects of updating class attribute w
        """
        n, d = X.shape
        if reg is None:
            self.w = np.linalg.solve(X.T @ X, X.T @ y)
        elif reg == 'l2':
            self.w = np.linalg.solve((X.T @ X + lam*np.eye(n)), X.T @ y)

    def predict(self, X):
        """
        External method for making predictions
        Inputs:
         - X: n x d array of observations to make predictions on
        Output:
         - an array of length n correponding to predictions for each obs
        """
        return X @ self.w
