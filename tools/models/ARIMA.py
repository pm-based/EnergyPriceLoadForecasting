"""
ARIMA model class
"""
from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import os
import shutil
import optuna
import matplotlib.pyplot as plt

from tools.evaluation_metrics import evaluation_metrics


class ARIMARegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],))
        logit = tf.keras.layers.Dense(self.settings['pred_horiz'],
                                      activation='linear',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['l1'],
                                                                                     l2=self.settings['l2'])
                                      )(x_in)
        output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        # Create model
        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss,
                           metrics=evaluation_metrics(self.settings['evaluation_metrics'])
                           )
    def build_model_input_from_series(self, x, col_names: List, pred_horiz: int):
        # get index of target and past features
        past_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['target'] in item or features_keys['past'] in item]

        # get index of const features
        const_col_idxs = [index for (index, item) in enumerate(col_names)
                          if features_keys['const'] in item]

        # get index of futu features
        futu_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['futu'] in item]

        # build conditioning variables for past features
        past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
        # build conditioning variables for futu features
        futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
        # build conditioning variables for cal features
        c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

        # return flattened input
        return np.concatenate(past_feat + futu_feat + c_feat, axis=1)

    def build_model_input_from_series(self, x, col_names: List, pred_horiz: int):
        # get index of target and past features
        past_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['target'] in item or features_keys['past'] in item]

        # get index of const features
        const_col_idxs = [index for (index, item) in enumerate(col_names)
                          if features_keys['const'] in item]

        # get index of futu features
        futu_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['futu'] in item]

        # build conditioning variables for past features
        past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
        # build conditioning variables for futu features
        futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
        # build conditioning variables for cal features
        c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

        # return flattened input
        return np.concatenate(past_feat + futu_feat + c_feat, axis=1)

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
                                              restore_best_weights=False)

        # Create folder to temporally store checkpoints
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor="val_loss", mode="min",
                                                save_best_only=True,
                                                save_weights_only=True, verbose=0)

        if pruning_call == None:
            callbacks = [es, cp]
        else:
            callbacks = [es, cp, pruning_call]

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=verbose)

        # Load best weights: do not use restore_best_weights from early stop since works only in case it stops training
        self.model.load_weights(checkpoint_path)
        # delete temporary folder
        shutil.rmtree(checkpoint_dir)
        return history

#def tune_hyperparameters(self, train_x, train_y, val_x, val_y, n_trials=100):
#    def objective(trial):
        # Suggest values for the hyperparameters
#        self.settings['l1'] = trial.suggest_float('l1', 1e-7, 1e-1, log=True)
#        self.settings['l2'] = trial.suggest_float('l2', 1e-7, 1e-1, log=True)
#        self.settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

        # Rebuild the model with the suggested hyperparameters
#        self.__build_model__(self.settings)

        # Fit the model and compute the validation loss
#        history = self.fit(train_x, train_y, val_x, val_y)

        # Use the validation loss as the objective to minimize
#        val_loss = history.history['val_loss'][-1]
#        return val_loss

    # Create an Optuna study object and optimize the hyperparameters
#    study = optuna.create_study(direction='minimize')
#   study.optimize(objective, n_trials=n_trials)

    # Return the best hyperparameters
#    return study.best_params