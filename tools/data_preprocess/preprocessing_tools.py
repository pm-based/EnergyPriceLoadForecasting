import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
import numpy as np


class Preprocessor:
    def __init__(self, preproc_configs=None, preproc_configs_file_path='None'):
        self.configs_file_path = preproc_configs_file_path
        if preproc_configs_file_path != 'None':
            with open(self.configs_file_path, 'r') as file:
                configs = json.load(file)
        elif preproc_configs is not None:
            configs = preproc_configs
        else:
            print('ERROR: No configs provided')
            return
        self.configs = configs
        self.methods = configs['methods']
        self.target = configs['data_config']['target']
        self.data = None
        self.StandardScaler = StandardScaler()
        self.RobustScaler = RobustScaler()
        self.ArcSinh = FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh)

    def load_data(self, df):
        self.data = df

    def load_data_from_path(self, path):
        self.data = pd.read_csv(path)

    def preprocess_data(self, pred_horiz):

        # NOTE! get pred_horiz from "data_config", in the class it should be self.data_config.pred_horiz

        for feature in self.methods:
            # Get the method to apply to the column
            method = self.methods[feature]
            df_feat = self.data[[feature]]
            if method == 'StandardScaler':
                self.StandardScaler.fit(df_feat[:-pred_horiz])  # Finds the mean and std to the data
                np_feat_scaled = self.StandardScaler.transform(df_feat)
                self.data[feature] = np_feat_scaled
            elif method == 'RobustScaler':
                self.RobustScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.RobustScaler.transform(df_feat)
                self.data[feature] = np_feat_scaled
            elif method == 'ArcSinh':
                self.ArcSinh.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.ArcSinh.transform(df_feat)
                self.data[feature] = np_feat_scaled
            elif method == 'Cyclical7':
                self.data[feature + '_sin'] = np.sin(2 * np.pi * df_feat / 7)/2 + 0.5
                self.data[feature + '_cos'] = np.cos(2 * np.pi * df_feat / 7)/2 + 0.5
                self.data.drop(feature, axis=1, inplace=True)  # Do I need this?
            elif method == 'Cyclical365':
                self.data[feature + '_sin'] = np.sin(2 * np.pi * df_feat / 365)/2 + 0.5
                self.data[feature + '_cos'] = np.cos(2 * np.pi * df_feat / 365)/2 + 0.5
                self.data.drop(feature, axis=1, inplace=True)  # Do I need this?
            # Add more methods here
            elif method == 'None':
                pass
            else:
                print('ERROR: Preprocessing method not found')

        return self.data

    def inverse_transform(self, model_configs, rescaled_PIs, ens_p, i):

        # Only transform the target column

        if self.methods[self.target] == 'StandardScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.methods[self.target] == 'RobustScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.RobustScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.methods[self.target] == 'ArcSinh':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.ArcSinh.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        # Add more methods here
        elif self.methods[self.target] == 'None':
            rescaled_PIs[model_configs['target_quantiles'][i]] = ens_p[:, 0]  # Not sure if this is correct
        else:
            print('ERROR: Inverse target transformation method not found')

        return rescaled_PIs
