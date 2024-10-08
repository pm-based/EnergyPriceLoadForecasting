"""
DNN model class
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os
import shutil
import matplotlib.pyplot as plt


class LSTMRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size']))
        x = tf.keras.layers.BatchNormalization()(x_in)
        for hl in range(self.settings['n_hidden_LSTM_layers'] - 1):
            x = tf.keras.layers.LSTM(self.settings['hidden_size'],
                                     # activation=self.settings['activation'],
                                     dropout=0.2, recurrent_dropout=0.2,
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                    l2=self.settings['l2']),
                                     return_sequences=True,
                                    )(x)
        x = (tf.keras.layers.LSTM(128,
                                  # activation=self.settings['activation'],
                                  return_sequences=False,
                                  # dropout=0.2, recurrent_dropout=0.2,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                 l2=self.settings['l2']),
                                  )(x))

        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          )(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        elif self.settings['PF_method'] == 'qr':
            out_size = len(self.settings['target_quantiles'])
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear')(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], out_size))(logit)
            # fix quintile crossing by sorting
            output = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(output)

        elif self.settings['PF_method'] == 'Normal':
            out_size = 2
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear')(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:])))(logit)

        elif self.settings['PF_method'] == 'JSU':
            out_size = 4
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear')(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    # clip_by_value keeps the parameters in a valid range
                    skewness=tf.clip_by_value(0.02 + t[..., :self.settings['pred_horiz']], -1.85, 1.8),
                    tailweight=tf.clip_by_value(
                        self.settings['JSU_tailweight'] +
                        t[..., self.settings['pred_horiz']:2 * self.settings['pred_horiz']],
                        1.2, 5.5),
                    loc=t[..., 2 * self.settings['pred_horiz']:3 * self.settings['pred_horiz']],
                    scale=tf.clip_by_value(
                        self.settings['JSU_Scale'] + tf.math.softplus(t[..., 3 * self.settings['pred_horiz']:]), 0.02,
                        0.3),
                    validate_args=True))(logit)

        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Create model
        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss)

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        # Convert the data into the input format using the internal converter
        train_x = self.build_model_input_from_series(x=train_x,
                                                     col_names=self.settings['x_columns_names'],
                                                     pred_horiz=self.settings['pred_horiz'])
        val_x = self.build_model_input_from_series(x=val_x,
                                                   col_names=self.settings['x_columns_names'],
                                                   pred_horiz=self.settings['pred_horiz'])

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=self.settings['patience'],
                                              restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tensorBoard_logs', histogram_freq=1)

        callbacks = [es, tensorboard_callback]

        if pruning_call is not None:
            callbacks.append(pruning_call)

        history = self.model.fit(train_x, train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=verbose)
        return history

    def predict(self, x):
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model.evaluate(x=x, y=y)

    def save_weights(self,path):
        return self.model.save_weights(path)

    def load_weights(self,path):
        return self.model.load_weights(path)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):

        return x

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        settings['hidden_size'] = 128  # trial.suggest_int('hidden_size', 64, 960, step=64)
        settings['n_hidden_LSTM_layers'] = 1  # trial.suggest_int('n_hidden_layers', 1, 3)
        settings['lr'] = 0.001  # trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        # settings['activation'] = 'tanh',
        settings['l1'] = 1e-3  # trial.suggest_float('l1', 1e-7, 1e-3, log=True)
        settings['l2'] = 1e-3  # trial.suggest_float('l2', 1e-7, 1e-3, log=True)
        settings['JSU_tailweight'] = trial.suggest_float('JSU_tailweight', 1e-2, 10, log=True)
        settings['JSU_Scale'] = trial.suggest_float('JSU_Scale', 1e-2, 0.8, log=True)

        return settings

    @staticmethod
    def get_hyperparams_searchspace():  # used only for grid search
        return {'hidden_size': [128, 512],
                'lr': [1e-4, 1e-3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):  # takes params from config file
        model_hyperparams = {
            'hidden_size': configs['hidden_size'],
            'n_hidden_LSTM_layers': configs['n_hidden_LSTM_layers'],
            'lr': configs['lr'],
            # 'activation': configs['activation'],
            'l1': configs['l1'],
            'l2': configs['l2'],
            'JSU_tailweight': configs['JSU_tailweight'],
            'JSU_Scale': configs['JSU_Scale']
        }
        return model_hyperparams
