#!/usr/bin/env python

import numpy as np

class softmax_regression():

    def __init__(self):

        self.W = None

    @staticmethod
    def softmax(arr):
        """
        Rowwise computation of softmax.
        Input:
         - arr: an array of dimension n x k (in this context X @ W.T)
        Output:
         - an array of the same dimension as input
        """
        a = arr.max(axis=1)
        ex_arr = np.exp(arr - a[:,None])
        return ex_arr / ex_arr.sum(axis=1)[:,None]

    @staticmethod
    def log_sum_exp(arr):
        """
        Rowwise computation of log-sum-exp.
        Input:
         - arr: an array of dimension n x k (in this context X @ W.T)
        Output:
         - an array of dimension n x 1
        """
        a = arr.max(axis=1)
        return a + np.log(np.sum(np.exp(arr - a[:,None]), axis=1))

    @staticmethod
    def one_hot(vec):
        """
        One hot encoder for response:
        Input:
         - vec: an n x 1 (response) vector
        Output:
         - an n x k one hot encoding of the input
        """
        # one hot encoding from:
        #  - https://stackoverflow.com/questions/29831489/numpy-1-hot-array
        one_hot = np.zeros((len(vec), len(np.unique(vec))))
        one_hot[np.arange(len(vec)), vec] = 1
        return one_hot

    @staticmethod
    def gradient_descent(f, grad_f, w0, alpha=0.005, eps=0.01):
        """
        Gradient Descent for fitting:
        Input:
         - f:      the function to be minimized (loss in this context)
         - grad_f: the gradient of the function to be minimized
         - w0:     an initial guess
         - alpha:  the learning rate (constant at this point)
         - eps:    the tolerance on the magnitude of the gradient
        Outputs:
         - a k x d dimensional array of weights w that minimizes f
        """
        w = w0
        g = grad_f(w)
        while np.linalg.norm(g) > eps:
            g = grad_f(w)
            w = w - alpha * grad_f(w)
            #print(np.linalg.norm(g))
        return w

    @staticmethod
    def loss(W, X, y):
        """
        Loss for Logistic Regression
        Inputs:
         - W: a k x d weight matrix where rows are the d weights for kth class
         - X: the n x d data matrix (n observations, d features)
         - y: the target classes
        Outputs:
         - a number that should mean something to someone
        """
        one_hot = softmax_regression.one_hot(y)

        W = W.reshape((len(np.unique(y)), X.shape[1]))
        xw = X @ W.T
        lse = softmax_regression.log_sum_exp(xw)
        return -np.sum(one_hot * (xw - lse[:,None]))

    @staticmethod
    def grad_loss(W, X, y):
        """
        Inputs:
         - W: k x d array of parameters
         - X: n x d design matrix
         - y: n x 1 target vector
        Output:
         - a k x d array of partial derivatives
        """
        one_hot = softmax_regression.one_hot(y)

        W = W.reshape((len(np.unique(y)), X.shape[1]))

        g = X.T @ (one_hot - softmax_regression.softmax(X @ W.T))
        return -g.T #-g.flatten(order='F')


    def fit(self, X, y):
        """
        Fit method:
        Input:
         - X: an n x d design matrix
         - y: a n x 1 target vector
        Ouput:
         - a k x d array of optimal weights
        """
        f = lambda w: softmax_regression.loss(w, X, y)
        grad_f = lambda w: softmax_regression.grad_loss(w, X, y)
        w0 = np.random.rand(len(np.unique(y))*X.shape[1])
        w0 = w0.reshape((len(np.unique(y)), X.shape[1]))
        self.W = softmax_regression.gradient_descent(f, grad_f, w0)
        return None


    def predict(self, X):
        """
        Multiclass soft prediction method:
        Input:
         - X: an n x d design matrix
        Output:
         - a n x k array of probabilistic predictions
           each observation gets k predicted probabilities

         - note: if you want hard predictions, use out.argmax(axis=1)
           where out is what this method returns
        """
        return softmax_regression.softmax(X @ self.W.T)



class knn():

    def __init__(self, k):

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit method for knn
        Input:
         - y: a n x 1 array of targets
         - X: a n x d array of observations
        Output:
         - No return, called for side effect of loading y, X into object
        """
        self.X_train = X
        self.y_train = y

    @staticmethod
    def _pred(obs, target, data, k):
        """
        Input:
         - obs:    a 1 x d array (single observation)
         - target: a n x 1 array of target (corresponds to data)
         - data:   a n x d array (all the observations)
         - k:      an integer specifying number of nearby points over which
                   to take mode
        Ouput:
         - mode value in target of k nearest neighbours of obs in data
        """
        dist = np.linalg.norm(data - obs, axis=1)
        close_k = target[np.argsort(dist)][:k]
        values, counts = np.unique(close_k, return_counts=True)
        return values[np.argmax(counts)]


    def predict(self, X):
        """
        Predict method for knn class
        Input:
         - X: an m x d array of observations we seek to make predictions on
        Output:
         - an m x 1 array of labels assigned to the m observations in X
        """
        one_d = lambda obs: knn._pred(obs, self.y_train, self.X_train, self.k)
        prediction = np.apply_along_axis(one_d, axis=1, arr=X)
        return prediction.astype('int')
