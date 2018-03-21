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
