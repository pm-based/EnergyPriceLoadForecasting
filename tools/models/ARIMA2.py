import numpy as np
import os
import shutil
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ARIMARegressor2:
    def __init__(self, settings, loss='mse'):
        self.settings = settings
        self.loss = loss
        self.model = None

    def fit(self, train_y):
        # Train the ARIMA model
        self.model = ARIMA(train_y, order=(self.settings['p'], self.settings['d'], self.settings['q']))
        self.fitted_model = self.model.fit()

    def predict(self, steps):
        # Predict using the ARIMA model
        if not self.model:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        return self.fitted_model.forecast(steps=steps)

    def evaluate(self, y_true):
        # Evaluate the model using the specified loss function
        predictions = self.predict(steps=len(y_true))
        if self.loss == 'mse':
            return mean_squared_error(y_true, predictions)
        elif self.loss == 'mae':
            return mean_absolute_error(y_true, predictions)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        settings['p'] = trial.suggest_int('p', 0, 5)
        settings['d'] = trial.suggest_int('d', 0, 2)
        settings['q'] = trial.suggest_int('q', 0, 5)
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'p': [0, 1, 2, 3, 4, 5],
                'd': [0, 1, 2],
                'q': [0, 1, 2, 3, 4, 5]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        model_hyperparams = {
            'p': configs['p'],
            'd': configs['d'],
            'q': configs['q']
        }
        return model_hyperparams

    def plot_diagnostics(self):
        # Plot diagnostics of the ARIMA model
        if not self.model:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        self.fitted_model.plot_diagnostics()
        plt.show()

    def print_model_summary(self):
        # Print the summary of the ARIMA model
        if not self.model:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        print(self.fitted_model.summary())
