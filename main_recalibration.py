"""
Main script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles, build_alpha_quantiles_map


#--------------------------------------------------------------------------------------------------------------------
def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score
#--------------------------------------------------------------------------------------------------------------------
def compute_Winkler_scores(y_true, alpha_list, pred_interval):
    """
       Utility function to compute the Winckler's score on the test results
       return: Winckler's scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for alpha in alpha_list:
        l_hat = pred_interval[alpha]['l']
        u_hat = pred_interval[alpha]['u']
        delta = np.subtract(u_hat, l_hat)
        score_i = delta + 2/(1-alpha) * (
                    np.maximum(np.subtract(l_hat, y_true), 0) + np.maximum(np.subtract(y_true, u_hat), 0))
        score.append(np.expand_dims(score_i, -1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score
#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'EM_price'
# Set Model setup to execute: point_ARX, point-DNN, QR-DNN, N-DNN
exper_setup = 'QR-DNN'

#---------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'compare_QR_N_JSU'

# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
# Plot train history flag
plot_train_history=False
plot_weights=False
print_weights_stats = False
#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

# Load dataset
dir_path = os.getcwd()
ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
ds.set_index(ds.columns[0], inplace=True)

#---------------------------------------------------------------------------------------------------------------------
# Instantiate recalibratione engine
PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                               data_configs=configs['data_config'],
                               model_configs=configs['model_config'])

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights,
                                               print_weights_stats=print_weights_stats)

#--------------------------------------------------------------------------------------------------------------------
# Compute pinball score
quantiles_levels = PrTSF_eng.model_configs['target_quantiles']
pred_steps = configs['model_config']['pred_horiz']

pinball_scores = compute_pinball_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)
# Print Pinball scores
print("Pinball Scores: ", pinball_scores)

#--------------------------------------------------------------------------------------------------------------------
# Compute Winkler's score
quantiles_levels = PrTSF_eng.model_configs['target_quantiles']
alpha_levels = PrTSF_eng.model_configs['target_alpha']
pred_interval = PrTSF_eng.model_configs['q_alpha_map']
pred_steps = configs['model_config']['pred_horiz']
#
Winkler_scores = compute_Winkler_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1, pred_steps),
                                        alpha_list=alpha_levels, pred_interval=pred_interval)

# Print Winkler scores
print("Winkler Scores: ", Winkler_scores)

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions, target=PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
print('Done!')
