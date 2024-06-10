"""
Main script to run the recalibration experiments
"""

# --Imports------------------------------------------------------------------------------------------------------------
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Suppress warnings
display_warnings = False
if not display_warnings:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
from tools.score_calculator import ScoreCalculator

# --Settings-----------------------------------------------------------------------------------------------------------
# Set the path to the experiment folder
experiment_type = 'JSU'
model_type      = 'BiLSTM_Parallel2'
experiment_id   = 'final_recalibration'

# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
load_weights = False
restore_weights_at_recalib = True

# Plot train history flag
plot_train_history = True
plot_weights = False
print_weights_stats = False
plot_quantiles_bool = True

# --Recalibration------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs = load_data_model_configs(experiment_type=experiment_type, model_type=model_type, experiment_id=experiment_id)
# Get the path for the scores
scores_path = os.path.join(os.getcwd(), 'experiments', experiment_type, model_type, experiment_id)

# Load dataset
dir_path = os.getcwd()
dataset = pd.read_csv(os.path.join(dir_path, 'data', configs['data_config'].dataset_name))
dataset.set_index(dataset.columns[0], inplace=True)

# Instantiate recalibration engine
PrTSF_eng = PrTsfRecalibEngine(dataset=dataset,
                               data_configs=configs['data_config'],
                               model_configs=configs['model_config'],
                               custom_preproc_configs=configs['custom_preprocessing_config'])

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode,
                                                    optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               path_history=scores_path,
                                               plot_weights=plot_weights,
                                               print_weights_stats=print_weights_stats,
                                               load_weights=load_weights,
                                               restore_weights_at_recalib=restore_weights_at_recalib)

# --Results------------------------------------------------------------------------------------------------------------
# Plot test predictions
PF_task_name = 'NetLoad'
if plot_quantiles_bool:
    plot_quantiles(test_predictions, target=PF_task_name, path_to_save=os.path.join(scores_path, 'quantiles'))

pred_steps = configs['model_config']['pred_horiz']
quantiles_levels = PrTSF_eng.model_configs['target_quantiles']

calculator = ScoreCalculator(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1, pred_steps),
                             pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                             quantiles_levels=quantiles_levels,
                             target_alpha=PrTSF_eng.model_configs['target_alpha'],
                             path_to_save=scores_path)

calculator.compute_pinball_scores()
calculator.compute_winkler_scores()
calculator.compute_delta_coverage()

calculator.display_scores(score_type='pinball', table=False, heatmap=True)
calculator.display_scores(score_type='winkler', table=False, heatmap=True)
calculator.display_scores(score_type='delta_coverage')
calculator.display_scores(score_type='rmse')

calculator.plot_scores_3d(score_type='pinball')
calculator.plot_scores_3d(score_type='winkler')

calculator.export_results()
calculator.export_scores(scores_path)

configs_to_save = load_data_model_configs(experiment_type=experiment_type, model_type=model_type, experiment_id=experiment_id)
calculator.add_to_score_table(path_to_table=os.path.join(os.getcwd(), 'experiments', 'scores_table'), configs=configs_to_save)

# ---------------------------------------------------------------------------------------------------------------------
print('Done!')