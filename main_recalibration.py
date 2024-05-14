"""
Main script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
# Suppress warnings
display_warnings = False
if not display_warnings:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles, build_alpha_quantiles_map
import matplotlib.pyplot as plt
from tools.score_calculator import ScoreCalculator
from mpl_toolkits.mplot3d import Axes3D

#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'EM_price'
# Set Model setup to execute: point_ARX, point-DNN, QR-DNN, N-DNN
exper_setup = 'QR-DNN'

#---------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'arcsinh_preprocessing'
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
# Plot train history flag
plot_train_history = False
plot_weights = False
print_weights_stats = False
plot_quantiles_bool = False
#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs = load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

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

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
if plot_quantiles_bool:
    plot_quantiles(test_predictions, target=PF_task_name)

calculator = ScoreCalculator(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1, pred_steps),
                             pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                             to_numpy().reshape(-1, pred_steps, len(quantiles_levels)), quantiles_levels=quantiles_levels)

calculator.compute_pinball_scores()
calculator.compute_winkler_scores()

calculator.display_scores(score_type='pinball', table=True, heatmap=True)
calculator.display_scores(score_type='winkler', table=True, heatmap=True)

calculator.plot_scores_3d(score_type='pinball')
calculator.plot_scores_3d(score_type='winkler')

calculator.export_scores("./")
calculator.export_scores("./")


#--------------------------------------------------------------------------------------------------------------------
print('Done!')
