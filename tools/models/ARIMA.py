"""
ARIMA model class
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import List
from ..evaluation_metrics import evaluation_metrics
from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

class ARIMARegressor:
    def __init__(self, settings):# p, q, d are integers
        self.order = settings
        self.model = None

    def fit(self, train):
        self.model = ARIMA(train, order=self.order)
        self.model_fit = self.model.fit(disp=0)

    def predict(self, start, end):
        if self.model_fit is None:
            raise Exception("The model is not fitted yet. Please call `fit` method before `predict`.")
        return self.model_fit.predict(start=start, end=end, dynamic=False)

    def get_hyperparams_trial(trial, settings):
        settings['p'] = trial.suggest_integer('p', 0, 10, log=True)
        settings['q'] = trial.suggest_integer('q', 0, 10, log=True)
        settings['d'] = trial.suggest_integer('d', 0, 10, log=True)
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'p': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'q': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'd': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        model_hyperparams = {
            'p': configs['p'],
            'q': configs['q'],
            'd': configs['d']
        }
        return model_hyperparams