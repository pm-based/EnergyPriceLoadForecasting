"""
Ensemble model
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license


import sys
from typing import List
import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt

from tools.models.DNN import DNNRegressor
from tools.models.ARX import ARXRegressor
from tools.models.LSTM import LSTMRegressor
from tools.models.BiLSTM import BiLSTMRegressor


def get_model_class_from_conf(conf):
    """
    Map the model class depending on the config name
    """
    if conf == 'ARX':
        model_class = ARXRegressor
    elif conf == 'DNN':
        model_class = DNNRegressor
    elif conf == 'LSTM':
        model_class = LSTMRegressor
    elif conf == 'BiLSTM':
        model_class = BiLSTMRegressor
    else:
        sys.exit('ERROR: unknown model_class')
    return model_class


def regression_model(settings, sample_x):
    """
    Wrapper to the regression model
    :param settings: model configurations, sample_x: input sample to derive the model input shape (first dimension has to be 1)
    :param sample_x: input sample, to derive the model input shape (first dimension has to be 1)
    :return: instantiated model
    """
    # Currently direct link to TF, Future dev pytorch
    return TensorflowRegressor(settings=settings, sample_x=sample_x)


class PinballLoss(keras.losses.Loss):
    def __init__(self, quantiles: List, name="pinball_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        loss = []
        for i, q in enumerate(self.quantiles):
            error = tf.subtract(y_true, y_pred[:, :, i])
            loss_q = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            loss.append(loss_q)
        L = tf.convert_to_tensor(loss)
        total_loss = tf.reduce_mean(L)
        return total_loss

    def get_config(self):
        return {
            "num_quantiles": self.quantiles,
            "name": self.name,
        }

class TensorflowRegressor():
    """
    Implementation of the Tenforflow regressor
    """
    def __init__(self, settings, sample_x):
        self.settings = settings
        self.x_columns_names = settings['x_columns_names']
        self.pred_horiz = settings['pred_horiz']

        tf.keras.backend.clear_session()
        # Map the loss to be used
        if settings['PF_method'] == 'qr':
            loss = PinballLoss(quantiles=settings['target_quantiles'])
        elif settings['PF_method'] == 'point':
            loss = 'mae'
        elif (settings['PF_method'] == 'Normal'
            or settings['PF_method'] == 'JSU'
        ):
            loss = lambda y, rv_y: -rv_y.log_prob(y) # Negative log-likelihood
        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Instantiate the model
        if settings['model_class']=='DNN':
            # get input size for the chosen model architecture
            settings['input_size']=DNNRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1]
            # Build the model architecture
            self.regressor = DNNRegressor(settings, loss)

        elif  settings['model_class']=='LSTM':
            # get input size for the chosen model architecture
            settings['input_size']=LSTMRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = LSTMRegressor(settings, loss)

        elif  settings['model_class']=='BiLSTM':
            # get input size for the chosen model architecture
            settings['input_size']=BiLSTMRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = BiLSTMRegressor(settings, loss)

        elif  settings['model_class']=='ARX':
            # get input size for the chosen model architecture
            settings['input_size']=ARXRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1]
            # Build the model architecture
            self.regressor = ARXRegressor(settings, loss)

        else:
            sys.exit('ERROR: unknown model_class')

        # Map handler to convert distributional output to quantiles or distribution parameters
        if (settings['PF_method'] == 'Normal'):
            self.output_handler = self.__pred_Normal_params__
        elif settings['PF_method'] == 'JSU':
            self.output_handler = self.__pred_JSU_params__
        else:
            self.output_handler =self.__quantiles_out__

    def predict(self, x):
        return self.output_handler(self.regressor.predict(x))

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None, plot_history=False, path_history=None):
        history = self.regressor.fit(train_x, train_y, val_x, val_y, verbose=0, pruning_call=None)
        if plot_history:
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='vali_loss')
            plt.grid()
            plt.legend()
            plt.show()
            if path_history is not None:
                plt.savefig(path_history)

    def evaluate(self, x, y):
        return self.regressor.evaluate(x=x, y=y)

    def save_weights(self,path):
        return self.regressor.save_weights(path=path)

    def load_weights(self,path):
        return self.regressor.load_weights(path=path)

    def plot_weights(self):
        self.regressor.plot_weights()

    def print_weights_stats(self):
        self.regressor.print_weights_stats()

    def __quantiles_out__(self, preds):
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(preds, axis=2)

    def __pred_Normal_params__(self, pred_dists: tfp.distributions):
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([loc,scale], axis=-1), axis=2)

    def __pred_JSU_params__(self, pred_dists: tfp.distributions):
        skewness = tf.expand_dims(pred_dists.skewness, axis=-1)
        tailweight = tf.expand_dims(pred_dists.tailweight, axis=-1)
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([skewness, tailweight, loc, scale], axis=-1), axis=2)



class Ensemble():
    """
    Tensorflow ensemble wrapper
    """
    def __init__(self, settings):
        # store configs for internal use
        self.settings = settings
        # map the methods to use for aggretation and quantile building depending on the configs
        if (self.settings['PF_method'] == 'point'):
            self.ensemble_aggregator = self.__aggregate_de_quantiles__
            self._build_test_PIs = self.__get_qr_PIs__
        elif (self.settings['PF_method'] == 'qr'):
            self.ensemble_aggregator = self.__aggregate_de_quantiles__
            self._build_test_PIs = self.__get_qr_PIs__
        elif (self.settings['PF_method'] == 'Normal'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_Normal_PIs__
        elif (self.settings['PF_method'] == 'JSU'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_JSU_PIs__
        else:
            sys.exit('ERROR: Ensemble config not supported!')

    def aggregate_preds(self, ens_comp_preds):
        # link function to the specific aggregator
        return self.ensemble_aggregator(ens_comp_preds=ens_comp_preds)

    def get_preds_test_quantiles(self, preds_test):
        # link function to the specific PI builder
        return self._build_test_PIs(preds_test=preds_test, settings=self.settings)

    @staticmethod
    def __aggregate_de__(ens_comp_preds):
        # aggregate by concatenation, for point a distributional settings
        return np.concatenate(ens_comp_preds, axis=2)

    @staticmethod
    def __aggregate_de_quantiles__(ens_comp_preds):
        # aggregate by a uniform vincentization
        return np.mean(np.concatenate(ens_comp_preds, axis=2), axis=2)

    @staticmethod
    def __get_qr_PIs__(preds_test, settings):
        # simply flatten in temporal dimension
        return preds_test.reshape(-1, preds_test.shape[-1])

    @staticmethod
    def __build_Normal_PIs__(preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.Normal(
                loc=preds_test[:,:,k,0],
                scale=preds_test[:,:,k,1]).sample(10000).numpy())
        return np.transpose(np.quantile(np.concatenate(pred_samples, axis=0),
                                        q=settings['target_quantiles'], axis=0),
                            axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))

    @staticmethod
    def __build_JSU_PIs__(preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.JohnsonSU(
                skewness=preds_test[:, :, k, 0],
                tailweight=preds_test[:, :, k, 1],
                loc=preds_test[:, :, k, 2],
                scale=preds_test[:, :, k, 3]).sample(100000).numpy())

            # check if it is Nan or Inf
            checkNanInf = np.isnan(pred_samples[-1]) | np.isinf(pred_samples[-1])
            if np.any(checkNanInf):
                print("Warning: pred_samples, the vector of samples of the JSU distribution contains NaN or Inf")

        computed_quantiles = np.quantile(np.concatenate(pred_samples, axis=0), q=settings['target_quantiles'], axis=0)
        return  np.transpose(computed_quantiles, axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))
