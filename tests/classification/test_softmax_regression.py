#!/usr/bin/env python

import numpy as np
from sklearn.datasets import load_digits
import pytest
import sys
sys.path.append('../') # not os agnostic


from mymlimp.classification import softmax_regression

X, y = load_digits(return_X_y=True)

def test_rowwise_softmax():
	"""
	Test that the softmax @staticmethod for softmax_regression class
	returns correct shape and row sum equals one
	"""
	soft_arr = softmax_regression.softmax(arr=X)
	row_sum_soft_arr = soft_arr.sum(axis=1)

	assert X.shape == soft_arr.shape
	assert np.allclose(row_sum_soft_arr, np.ones(soft_arr.shape[0]))

def test_rowwise_log_sum_exp():
	"""
	Test that log_sum_exp @staticmethod for softmax_regression class
	returns correct shape
	"""
	lse_arr = softmax_regression.log_sum_exp(arr=X)
	assert X.shape[0] == lse_arr.shape[0]

def test_gradient_descent():
	"""
	Test that gradient descent is returning array of correct shape
	and that it minimizes the norm of gradient on toy example
	"""
	f = lambda x: x**2
	grad_f = lambda x: 2*x
	w0 = 5.0
	eps = 0.001
	gd = softmax_regression.gradient_descent(f, grad_f, w0, eps=eps)
	assert type(gd) == type(w0)
	assert np.isclose(gd, 0, atol=eps)

def test_loss():
	"""
	Test that softmax loss returns an objecct of type float
	"""
	k = len(np.unique(y))
	n, d = X.shape
	W = np.random.random((k,d))
	loss = softmax_regression.loss(W, X, y)
	assert isinstance(loss, float)

def test_grad_loss():
	"""
	Test that gradient of softmax loss returns array of correct shape
	"""
	k = len(np.unique(y))
	n, d = X.shape
	W = np.random.random((k,d))
	grad_loss = softmax_regression.grad_loss(W, X, y)
	assert grad_loss.shape == W.shape

def test_fit_predict():
	"""
	Tests that fit and predict methods return an array of the correct
	dimension that has a row sum to one
	"""
	model = softmax_regression()
	model.fit(X,y)
	pred = model.predict(X)
	assert pred.shape == (X.shape[0], len(np.unique(y)))
	assert np.allclose(pred.sum(axis=1), np.ones(X.shape[0]))
