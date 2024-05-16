import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from typing import List
from tools.data_utils import features_keys


class HybridARXDNNRegressor:
    def __init__(self, settings, loss_arx, loss_dnn):
        self.settings = settings
        self.arx_model = ARXRegressor(settings, loss_arx)
        self.dnn_model = DNNModel(settings, loss_dnn)

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        # Fit ARX model
        history_arx = self.arx_model.fit(train_x, train_y, val_x, val_y, verbose, pruning_call)

        # Predict with ARX model to get residuals
        y_pred_arx_train = self.arx_model.predict(train_x)
        residuals_train = train_y - np.squeeze(y_pred_arx_train.numpy())

        y_pred_arx_val = self.arx_model.predict(val_x)
        residuals_val = val_y - np.squeeze(y_pred_arx_val.numpy())



        # Fit DNN model on residuals
        history_dnn = self.dnn_model.fit(train_x, train_y, val_x, residuals_val, verbose, pruning_call)

        return history_dnn#, history_dnn

    def predict(self, x):
        # Predict with ARX model
        y_pred_arx = self.arx_model.predict(x)

        # Predict residuals with DNN model
        y_pred_dnn = self.dnn_model.predict(x).mean()

        # Combine predictions
        final_predictions = y_pred_arx + y_pred_dnn

        return final_predictions

    def evaluate(self, x, y):
        # Evaluate ARX model
        arx_eval = self.arx_model.evaluate(x, y)

        # Predict with ARX model to get residuals
        y_pred_arx = self.arx_model.predict(x)
        residuals = y - y_pred_arx

        # Evaluate DNN model on residuals
        dnn_eval = self.dnn_model.evaluate(x, residuals)

        return dnn_eval#, dnn_eval

    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
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

class ARXRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],))
        logit = tf.keras.layers.Dense(self.settings['pred_horiz'],
                                      activation='linear',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.settings['ARX_l1'],
                                                                                     l2=self.settings['ARX_l2'])
                                      )(x_in)
        output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss)

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        train_x = self.build_model_input_from_series(train_x, self.settings['x_columns_names'],
                                                     self.settings['pred_horiz'])
        val_x = self.build_model_input_from_series(val_x, self.settings['x_columns_names'], self.settings['pred_horiz'])

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.settings['patience'],
                                              restore_best_weights=False)
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", mode="min",
                                                save_best_only=True, save_weights_only=True, verbose=0)
        callbacks = [es, cp] if pruning_call is None else [es, cp, pruning_call]

        history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'], callbacks=callbacks, verbose=verbose)
        self.model.load_weights(checkpoint_path)
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        x = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        x = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        return self.model.evaluate(x, y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
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


class DNNModel:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],))
        x_in = tf.keras.layers.BatchNormalization()(x_in)
        x = tf.keras.layers.Dense(self.settings['hidden_size'], activation=self.settings['activation'])(x_in)
        for _ in range(self.settings['n_hidden_layers'] - 1):
            x = tf.keras.layers.Dense(self.settings['hidden_size'], activation=self.settings['activation'])(x)
        out_size = 4
        logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size, activation='linear')(x)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.JohnsonSU(
                skewness=t[..., :self.settings['pred_horiz']],
                tailweight=t[..., self.settings['pred_horiz']:2 * self.settings['pred_horiz']],
                loc=t[..., 2 * self.settings['pred_horiz']:3 * self.settings['pred_horiz']],
                scale=1e-3 + tf.math.softplus(t[..., 3 * self.settings['pred_horiz']:])))(logit)

        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']), loss=loss)

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.settings['patience'],
                                              restore_best_weights=False)
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", mode="min",
                                                save_best_only=True, save_weights_only=True, verbose=0)
        callbacks = [es, cp] if pruning_call is None else [es, cp, pruning_call]

        history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'], callbacks=callbacks, verbose=verbose)
        self.model.load_weights(checkpoint_path)
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        return self.model(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
