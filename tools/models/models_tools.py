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
from tools.models.LSTM_Autoencoder import LSTM_Autoencoder_Regressor
from tools.models.BiLSTM_Parallel2 import BiLSTM_Parallel2Regressor
from tools.models.CNNLSTM import CNNLSTMRegressor
from tools.models.CNN import CNNRegressor
from tools.models.CNN2 import CNN2Regressor


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
    elif conf == 'LSTM_A':
        model_class = LSTM_Autoencoder_Regressor
    elif conf == 'BiLSTM_Parallel2':
        model_class = BiLSTM_Parallel2Regressor
    elif conf == 'CNNLSTM':
        model_class = CNNLSTMRegressor
    elif conf == 'CNN':
        model_class = CNNRegressor
    elif conf == 'CNN2':
        model_class = CNN2Regressor
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
        elif settings['PF_method'] == 'Normal' or settings['PF_method'] == 'JSU':
            loss = lambda y, rv_y: -rv_y.log_prob(y)  # Negative log-likelihood
        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Instantiate the model
        if settings['model_class'] == 'DNN':
            # get input size for the chosen model architecture
            settings['input_size'] = DNNRegressor.build_model_input_from_series(x=sample_x,
                                                                                col_names=self.x_columns_names,
                                                                                pred_horiz=self.pred_horiz).shape[1]
            # Build the model architecture
            self.regressor = DNNRegressor(settings, loss)

        elif settings['model_class'] == 'LSTM':
            # get input size for the chosen model architecture
            settings['input_size'] = LSTMRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = LSTMRegressor(settings, loss)

        elif  settings['model_class'] == 'BiLSTM':
            # get input size for the chosen model architecture
            settings['input_size'] = BiLSTMRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = BiLSTMRegressor(settings, loss)

        elif  settings['model_class'] == 'ARX':
            # get input size for the chosen model architecture
            settings['input_size'] = ARXRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1]
            # Build the model architecture
            self.regressor = ARXRegressor(settings, loss)

        elif  settings['model_class'] == 'LSTM_A':
            # get input size for the chosen model architecture
            settings['input_size'] = LSTM_Autoencoder_Regressor.build_model_input_from_series(x=sample_x,
                                                                                              col_names=self.x_columns_names,
                                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = LSTM_Autoencoder_Regressor(settings, loss)

        elif  settings['model_class'] == 'BiLSTM_Parallel2':
            # get input size for the chosen model architecture
            settings['input_size'] = BiLSTM_Parallel2Regressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = BiLSTM_Parallel2Regressor(settings, loss)

        elif  settings['model_class'] == 'CNNLSTM':
            # get input size for the chosen model architecture
            settings['input_size'] = CNNLSTMRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = CNNLSTMRegressor(settings, loss)

        elif  settings['model_class'] == 'CNN':
            # get input size for the chosen model architecture
            settings['input_size'] = CNNRegressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = CNNRegressor(settings, loss)

        elif  settings['model_class'] == 'CNN2':
            # get input size for the chosen model architecture
            settings['input_size'] = CNN2Regressor.build_model_input_from_series(x=sample_x,
                                                                              col_names=self.x_columns_names,
                                                                              pred_horiz=self.pred_horiz).shape[1:]
            # Build the model architecture
            self.regressor = CNN2Regressor(settings, loss)

        else:
            sys.exit('ERROR: unknown model_class')

        # Map handler to convert distributional output to quantiles or distribution parameters
        if settings['PF_method'] == 'Normal':
            self.output_handler = self.__pred_Normal_params__
        elif settings['PF_method'] == 'JSU':
            self.output_handler = self.__pred_JSU_params__
        elif settings['PF_method'] == 'tStudent':
            self.output_handler = self.__pred_tStudent_params__
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
        return tf.expand_dims(tf.concat([loc, scale], axis=-1), axis=2)

    def __pred_JSU_params__(self, pred_dists: tfp.distributions):
        skewness = tf.expand_dims(pred_dists.skewness, axis=-1)
        tailweight = tf.expand_dims(pred_dists.tailweight, axis=-1)
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([skewness, tailweight, loc, scale], axis=-1), axis=2)

    def __pred_tStudent_params__(self, pred_dists: tfp.distributions):
        df = tf.expand_dims(pred_dists.df, axis=-1)
        loc = tf.expand_dims(pred_dists.loc, axis=-1)
        scale = tf.expand_dims(pred_dists.scale, axis=-1)
        # Expand dimension to enable concat in ensemble
        return tf.expand_dims(tf.concat([df, loc, scale], axis=-1), axis=2)



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
        elif (self.settings['PF_method'] == 'tStudent'):
            self.ensemble_aggregator = self.__aggregate_de__
            self._build_test_PIs = self.__build_Normal_PIs__
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
                scale=preds_test[:, :, k, 3]).sample(10000).numpy())

            # check if it is Nan or Inf
            checkNanInf = np.isnan(pred_samples[-1]) | np.isinf(pred_samples[-1])
            if np.any(checkNanInf):
                print("Warning: pred_samples, the vector of samples of the JSU distribution contains NaN or Inf")

        computed_quantiles = np.quantile(np.concatenate(pred_samples, axis=0), q=settings['target_quantiles'], axis=0)
        return  np.transpose(computed_quantiles, axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))

    def __build_tStudent_PIs__(self, preds_test, settings):
        # for each de component, sample, aggregate samples and compute quantiles
        pred_samples = []
        for k in range(preds_test.shape[2]):
            pred_samples.append(tfd.StudentT(
                df=1 + tf.math.softplus(0.05 * preds_test[:, :, k, 0]),
                loc=preds_test[:, :, k, 1],
                scale=1e-3 + 3 * tf.math.softplus(0.05 * preds_test[:, :, k, 2])).sample(10000).numpy())
        return np.transpose(np.quantile(np.concatenate(pred_samples, axis=0),
                                        q=settings['target_quantiles'], axis=0),
                            axes=(1, 2, 0)).reshape(-1, len(settings['target_quantiles']))

    def get_JSU_params(self, pred, trial=None):
        """
        Extracts the JSU parameters from the predictions
        """
        JSU_params = {}

        # Calculate means
        JSU_params['mean_skewness'] = np.mean(pred[:, :, 0, 0])
        JSU_params['mean_tailweight'] = np.mean(pred[:, :, 0, 1])
        JSU_params['mean_loc'] = np.mean(pred[:, :, 0, 2])
        JSU_params['mean_scale'] = np.mean(pred[:, :, 0, 3])

        # Calculate maxima
        JSU_params['max_skewness'] = np.max(pred[:, :, 0, 0])
        JSU_params['max_tailweight'] = np.max(pred[:, :, 0, 1])
        JSU_params['max_loc'] = np.max(pred[:, :, 0, 2])
        JSU_params['max_scale'] = np.max(pred[:, :, 0, 3])

        # Calculate minima
        JSU_params['min_skewness'] = np.min(pred[:, :, 0, 0])
        JSU_params['min_tailweight'] = np.min(pred[:, :, 0, 1])
        JSU_params['min_loc'] = np.min(pred[:, :, 0, 2])
        JSU_params['min_scale'] = np.min(pred[:, :, 0, 3])

        if trial is not None:
            # Set user attributes for means
            trial.set_user_attr("mean skewness", JSU_params['mean_skewness'].astype(np.float64))
            trial.set_user_attr("mean tailweight", JSU_params['mean_tailweight'].astype(np.float64))
            trial.set_user_attr("mean loc", JSU_params['mean_loc'].astype(np.float64))
            trial.set_user_attr("mean scale", JSU_params['mean_scale'].astype(np.float64))

            # Set user attributes for maxima
            trial.set_user_attr("max skewness", JSU_params['max_skewness'].astype(np.float64))
            trial.set_user_attr("max tailweight", JSU_params['max_tailweight'].astype(np.float64))
            trial.set_user_attr("max loc", JSU_params['max_loc'].astype(np.float64))
            trial.set_user_attr("max scale", JSU_params['max_scale'].astype(np.float64))

            # Set user attributes for minima
            trial.set_user_attr("min skewness", JSU_params['min_skewness'].astype(np.float64))
            trial.set_user_attr("min tailweight", JSU_params['min_tailweight'].astype(np.float64))
            trial.set_user_attr("min loc", JSU_params['min_loc'].astype(np.float64))
            trial.set_user_attr("min scale", JSU_params['min_scale'].astype(np.float64))

            return trial

        return JSU_params
