#!/usr/bin/env python

import numpy as np
from sklearn.datasets import load_digits
import pytest
import sys
sys.path.append('../') # not os agnostic


from mymlimp.classification import knn

X, y = load_digits(return_X_y=True)

def test_fit():
    """
    Tests that the fir method loads data correctly
    """
    model = knn(k=3)
    model.fit(X,y)
    assert model.X_train.shape == X.shape
    assert model.y_train.shape == y.shape

def test_internal_prediction():
    """
    Test the return of the interval single observation predict @staticmethod
    """
    obs = X[0,:]
    k = 3
    prediction = knn._pred(obs, y, X, k)
    assert prediction.shape == ()

def test_external_prediction():
    """
    Tests the shape and contents of external prediction method
    """
    model = knn(k=3)
    model.fit(X,y)
    prediction = model.predict(X)
    assert prediction.shape == y.shape
    assert len(np.unique(prediction)) <= len(np.unique(y))
