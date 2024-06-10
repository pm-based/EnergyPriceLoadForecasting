import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from pyts.decomposition import SingularSpectrumAnalysis
from scipy import stats
import numpy as np


class Preprocessor:
    def __init__(self, custom_preproc_configs: dict = None, preproc_configs_file_path='None'):
        self.configs_file_path = preproc_configs_file_path
        if preproc_configs_file_path != 'None':
            with open(self.configs_file_path, 'r') as file:
                configs = json.load(file)
        elif custom_preproc_configs is not None:
            configs = custom_preproc_configs
        else:
            print('ERROR: No custom preprocessing configs provided')
            return
        self.features_methods = configs['features']
        self.target_methods = configs['target']

        self.StandardScaler = StandardScaler()
        self.RobustScaler = RobustScaler()
        self.ArcSinh = FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh)
        self.MinMaxScaler = MinMaxScaler()
        self.MaxAbsScaler = MaxAbsScaler()
        self.Log1p = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        self.TargetScaler = None

    def preprocess_data(self, data, pred_horiz):
        """
        Preprocess the data according to the methods specified in configs
        """
        processed_data = data.copy()
        all_methods = self.features_methods.copy()
        all_methods.update(self.target_methods)
        # The target is put at the end, so that the inverse transform is applied to the scaler fit to the target

        for feature in all_methods:
            method = all_methods[feature]
            df_feat = data[[feature]]
            if method == 'StandardScaler':
                self.StandardScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.StandardScaler.transform(df_feat)
                processed_data[feature] = np_feat_scaled
            elif method == 'RobustScaler':
                self.RobustScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.RobustScaler.transform(df_feat)
                processed_data[feature] = np_feat_scaled
            elif method == 'ArcSinh':
                self.ArcSinh.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.ArcSinh.transform(df_feat)
                processed_data[feature] = np_feat_scaled
            elif method == 'MinMaxScaler':
                self.MinMaxScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.MinMaxScaler.transform(df_feat)
                processed_data[feature] = np_feat_scaled
            elif method == 'MaxAbsScaler':
                self.MaxAbsScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.MaxAbsScaler.transform(df_feat)
                processed_data[feature] = np_feat_scaled
            elif method == 'Log1pStandardScaler':
                self.Log1p.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.Log1p.transform(df_feat)
                self.StandardScaler.fit(np_feat_scaled[:-pred_horiz])
                np_feat_scaled = self.StandardScaler.transform(np_feat_scaled)
                processed_data[feature] = np_feat_scaled
            elif method == 'LogStandardScaler':
                self.StandardScaler.fit(np.log(df_feat[:-pred_horiz]))
                np_feat_scaled = self.StandardScaler.transform(np.log(df_feat))
                processed_data[feature] = np_feat_scaled
            elif method == 'BoxCox1e-6':
                df_feat = df_feat + 1e-6
                df_feat, _ = stats.boxcox(df_feat.squeeze())
                self.StandardScaler.fit(df_feat[:-pred_horiz].reshape(-1, 1))
                np_feat_scaled = self.StandardScaler.transform(df_feat.reshape(-1, 1))
                processed_data[feature] = np_feat_scaled
            elif method == 'BoxCox1e-8':
                df_feat = df_feat + 1e-8
                df_feat, _ = stats.boxcox(df_feat.squeeze())
                self.StandardScaler.fit(df_feat[:-pred_horiz].reshape(-1, 1))
                np_feat_scaled = self.StandardScaler.transform(df_feat.reshape(-1, 1))
                processed_data[feature] = np_feat_scaled
            elif method == 'BoxCoxLog':
                df_feat = df_feat + 1e3
                df_feat, _ = stats.boxcox(df_feat.squeeze())
                self.StandardScaler.fit(np.log(df_feat[:-pred_horiz]).reshape(-1, 1))
                np_feat_scaled = self.StandardScaler.transform(np.log(df_feat).reshape(-1, 1))
                processed_data[feature] = np_feat_scaled
            elif method == 'Cyclical24':
                processed_data[feature + '_sin'] = np.sin(2 * np.pi * df_feat / 24)/2 + 0.5
                processed_data[feature + '_cos'] = np.cos(2 * np.pi * df_feat / 24)/2 + 0.5
                processed_data.drop(feature, axis=1, inplace=True)
            elif method == 'Cyclical7':
                processed_data[feature + '_sin'] = np.sin(2 * np.pi * df_feat / 7)/2 + 0.5
                processed_data[feature + '_cos'] = np.cos(2 * np.pi * df_feat / 7)/2 + 0.5
                processed_data.drop(feature, axis=1, inplace=True)
            elif method == 'Cyclical365':
                processed_data[feature + '_sin'] = np.sin(2 * np.pi * df_feat / 365)/2 + 0.5
                processed_data[feature + '_cos'] = np.cos(2 * np.pi * df_feat / 365)/2 + 0.5
                processed_data.drop(feature, axis=1, inplace=True)
            elif method == 'SSA':
                ssa = SingularSpectrumAnalysis(window_size=150)

                # standard scaler first
                self.StandardScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.StandardScaler.transform(df_feat)

                #ssa part
                np_feat_scaled = np.array(np_feat_scaled)
                ssa.fit(np_feat_scaled[:-pred_horiz].reshape(1, -1))
                np_feat_ssa = ssa.transform(np_feat_scaled.reshape(1, -1))
                np_feat_ssa_reconstructed = np.sum(np_feat_ssa, axis=1).reshape(-1, 1)

                processed_data[feature] = np_feat_ssa_reconstructed
            elif method == 'SSA_decomp':
                ssa = SingularSpectrumAnalysis(window_size=5)

                # standard scaler first
                self.StandardScaler.fit(df_feat[:-pred_horiz])
                np_feat_scaled = self.StandardScaler.transform(df_feat)
                processed_data[feature] = np_feat_scaled

                # ssa part
                np_feat_scaled = np.array(np_feat_scaled)
                ssa.fit(np_feat_scaled[:-pred_horiz].reshape(1, -1))
                np_feat_ssa = ssa.transform(np_feat_scaled.reshape(1, -1))

                for n in range(5):
                    processed_data[feature + '_SSA_' + str(n)] = np_feat_ssa[:, n, :].reshape(-1, 1)
            # Add more methods here
            elif method == 'None':
                pass
            else:
                print('ERROR: Preprocessing method not recognized')

        return processed_data

    def inverse_transform(self, model_configs, rescaled_PIs, ens_p, i):
        """
        Inverse transform the predictions. Only for the target
        """
        keys = self.target_methods.keys()
        target = list(keys)[0]

        if self.target_methods[target] == 'StandardScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'RobustScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.RobustScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'ArcSinh':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.ArcSinh.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'MinMaxScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.MinMaxScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'MaxAbsScaler':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.MaxAbsScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'Log1p':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.Log1p.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'SSA':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(ens_p[:, i:i + 1])[:, 0]
        elif self.target_methods[target] == 'SSA_decomp':
            rescaled_PIs[model_configs['target_quantiles'][i]] = self.StandardScaler.inverse_transform(
                ens_p[:, i:i + 1])[:, 0]
        # Add more methods here
        elif self.target_methods[target] == 'None':
            rescaled_PIs[model_configs['target_quantiles'][i]] = ens_p[:, i]
        else:
            print('ERROR: Inverse target transformation method not found')

        return rescaled_PIs
