#! usr/bin/env python

import numpy as np
import pytest
import sys
sys.path.append('../')) # not os agnostic
from sklearn.datasets import load_linnerud

from mymlimp.regression import linear_model

X, y = load_linnerud(return_X_y=True)


def test_fit():
    """
    Test fit method produces correct shape weights w
    """
    pass

def test_predict():
    """
    Test predict method produces correct shape array of predictions
    """
