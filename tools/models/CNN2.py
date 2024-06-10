"""
CNN model class
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


class CNN2Regressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=self.settings['input_size'])
        x_in = tf.keras.layers.BatchNormalization()(x_in)
        x = (tf.keras.layers.Conv1D(64, kernel_size = 5,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                   l2=self.settings['l2'])
                                  #activation=self.settings['activation'],
                                  )(x_in))
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(128, kernel_size = 5,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                  l2=self.settings['l2'])
                                    #activation=self.settings['activation'],
                                    )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(256, kernel_size = 5,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                  l2=self.settings['l2'])
                                    #activation=self.settings['activation'],
                                    )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.LSTM(256,
                                 # activation=self.settings['activation'],
                                 return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                l2=self.settings['l2']))(x)
        x = tf.keras.layers.LSTM(128,
                                  #activation=self.settings['activation'],
                                 return_sequences=False,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2']))(x)

        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          )(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        elif self.settings['PF_method'] == 'qr':
            out_size = len(self.settings['target_quantiles'])
            logit = tf.keras.layers.Conv1D(self.settings['pred_horiz'] * out_size, kernel_size = 1,
                                          activation='linear')(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], out_size))(logit)
            #fix quintile crossing by sorting
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
                    skewness=t[..., :self.settings['pred_horiz']],
                    tailweight=1e-3 + tf.nn.relu(self.settings['JSU_tailweight'] +
                                                 t[..., self.settings['pred_horiz']:2 * self.settings['pred_horiz']]),
                    loc=t[..., 2 * self.settings['pred_horiz']:3 * self.settings['pred_horiz']],
                    scale=1e-3 + tf.nn.relu(self.settings['JSU_Scale'] +
                                            tf.math.softplus(t[..., 3 * self.settings['pred_horiz']:])),
                    validate_args=True))(logit)

        elif self.settings['PF_method'] == 'Gamma':
            out_size = 2
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear')(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.Gamma(
                    concentration=t[..., :self.settings['pred_horiz']],
                    rate=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:])))(logit)

        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Create model
        self.model= tf.keras.Model(inputs=[x_in], outputs=[output])
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
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=15,
            min_lr=0.0001
        )

        # Definisci il percorso dove salvare i pesi
        checkpoint_filepath = './checkpoints/model_epoch_{epoch:02d}.h5'

        # Configura il callback ModelCheckpoint
        cp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,  # Percorso per salvare i pesi
            save_weights_only=True,  # Salva solo i pesi del modello
            save_freq='epoch',  # Salva ogni epoca
            monitor='val_loss',  # Monitora la loss sul validation set
            mode='min',  # Minimizza la loss
            save_best_only=True)

        # Configura i callbacks includendo sempre tensorboard_callback
        callbacks = [es, reduce_lr, cp]
        self.model.summary()
        # Aggiungi pruning_call ai callbacks se presente
        if pruning_call is not None:
            callbacks.append(pruning_call)

        callbacks = [es, reduce_lr]

        self.model.summary()

        if pruning_call is not None:
            callbacks.append(pruning_call)

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=2)

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
        settings['hidden_size'] = trial.suggest_int('hidden_size', 64, 960, step=64)
        settings['n_hidden_LSTM_layers'] = 2  # trial.suggest_int('n_hidden_layers', 1, 3)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        settings['activation'] = 'softplus',
        settings['l1'] = trial.suggest_float('l1', 1e-7, 1e-3, log=True)
        settings['l2'] = trial.suggest_float('l2', 1e-7, 1e-3, log=True)
        settings['JSU_tailweight'] = trial.suggest_float('JSU_tailweight', 1e-2, 10, log=True)
        settings['JSU_Scale'] = trial.suggest_float('JSU_Scale', 1e-2, 0.8, log=True)

        return settings

    @staticmethod
    def get_hyperparams_searchspace(): # used only for grid search
        return {'hidden_size': [128, 512],
                'lr': [1e-4, 1e-3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs): # takes params from config file
        model_hyperparams = {
            'hidden_size': configs['hidden_size'],
            'n_hidden_LSTM_layers': configs['n_hidden_LSTM_layers'],
            'lr': configs['lr'],
            'activation': configs['activation'],
            'l1': configs['l1'],
            'l2': configs['l2'],
            'JSU_tailweight': configs['JSU_tailweight'],
            'JSU_Scale': configs['JSU_Scale']
        }
        return model_hyperparams