import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SARIMAXRegressor:
    def __init__(self, settings, loss='mse'):
        self.settings = settings
        self.loss = loss
        self.model = None

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        endog = train_x[:, 2]
        exog = train_x[:, 3:len(train_x[0])]
        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
        seasonal_order = (self.settings.get('P', 0), self.settings.get('D', 0), self.settings.get('Q', 0), self.settings.get('S', 0))
        self.model = sm.tsa.SARIMAX(endog, exog,
                                     order=(self.settings['p'], self.settings['d'], self.settings['q']),
                                     seasonal_order=seasonal_order,)
        self.fitted_model = self.model.fit(disp=verbose)

    def predict(self, steps, exog=None):
        if not self.fitted_model:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
        return self.fitted_model.forecast(steps=steps, exog=exog)

    def evaluate(self, y_true, exog=None):
        predictions = self.predict(steps=len(y_true), exog=exog)
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
        settings['P'] = trial.suggest_int('P', 0, 2)
        settings['D'] = trial.suggest_int('D', 0, 1)
        settings['Q'] = trial.suggest_int('Q', 0, 2)
        settings['S'] = trial.suggest_int('S', 0, 12)
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {
            'p': [0, 1, 2, 3, 4, 5],
            'd': [0, 1, 2],
            'q': [0, 1, 2, 3, 4, 5],
            'P': [0, 1, 2],
            'D': [0, 1],
            'Q': [0, 1, 2],
            'S': [0, 6, 12]
        }

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        model_hyperparams = {
            'p': configs['p'],
            'd': configs['d'],
            'q': configs['q'],
            'P': configs.get('P', 0),
            'D': configs.get('D', 0),
            'Q': configs.get('Q', 0),
            'S': configs.get('S', 0)
        }
        return model_hyperparams
