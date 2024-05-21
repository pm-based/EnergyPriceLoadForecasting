import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from pyts.decomposition import SingularSpectrumAnalysis
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
        self.TargetScaler = None

    def load_data(self, df):
        self.data = df

    def load_data_from_path(self, path):
        self.data = pd.read_csv(path)
    #TODO: GESTIRE IL FATTO CHE ORA IL TARGET DEVE ESSERE L'ULTIMO NEL JSON.
    def preprocess_data(self, pred_horiz):

        # NOTE! get pred_horiz from "data_config", in the class it should be self.data_config.pred_horiz

        for feature in self.methods:  # TODO: now it depends on the order of the target method in the json file. Change it so that it is always the last one
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
            elif method == 'SSA':
                ssa = SingularSpectrumAnalysis(window_size=10)

                # standar scaler part
                self.StandardScaler.fit(df_feat[:-pred_horiz])  # Finds the mean and std to the data
                np_feat_scaled = self.StandardScaler.transform(df_feat)

                #ssa part
                np_feat_scaled = np.array(np_feat_scaled)
                ssa.fit(np_feat_scaled[:-pred_horiz].reshape(1, -1))
                np_feat_ssa = ssa.transform(np_feat_scaled.reshape(1, -1))
                np_feat_ssa_reconstruct = np.sum(np_feat_ssa, axis=1).reshape(-1,1)

                self.data[feature] = np_feat_ssa_reconstruct
            elif method == 'SSA_decomp':
                ssa = SingularSpectrumAnalysis(window_size=5)

                # standar scaler part
                self.StandardScaler.fit(df_feat[:-pred_horiz])  # Finds the mean and std to the data
                np_feat_scaled = self.StandardScaler.transform(df_feat)
                self.data[feature] = np_feat_scaled

                # ssa part
                np_feat_scaled = np.array(np_feat_scaled)
                ssa.fit(np_feat_scaled[:-pred_horiz].reshape(1, -1))
                np_feat_ssa = ssa.transform(np_feat_scaled.reshape(1, -1))
                #np_feat_ssa_reconstruct = np.sum(np_feat_ssa, axis=1).reshape(-1, 1)

                self.data["SSA_0"] = np_feat_ssa[:,0,:].reshape(-1, 1)
                self.data["SSA_1"] = np_feat_ssa[:,1,:].reshape(-1, 1)
                self.data["SSA_2"] = np_feat_ssa[:,2,:].reshape(-1, 1)
                self.data["SSA_3"] = np_feat_ssa[:,3,:].reshape(-1, 1)
                self.data["SSA_4"] = np_feat_ssa[:,4,:].reshape(-1, 1)
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
        elif self.methods[self.target] == 'SSA':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.methods[self.target] == 'SSA_decomp':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(
                ens_p[:, i:i + 1])[:, 0]
        # Add more methods here
        elif self.methods[self.target] == 'None':
            rescaled_PIs[model_configs['target_quantiles'][i]] = ens_p[:, i]  # Not sure if this is correct
        else:
            print('ERROR: Inverse target transformation method not found')

        return rescaled_PIs
