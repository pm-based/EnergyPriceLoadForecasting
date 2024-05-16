import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os


class ScoreCalculator:
    def __init__(self, y_true, pred_quantiles, quantiles_levels):
        self.y_true = y_true
        self.pred_quantiles = pred_quantiles
        self.quantiles_levels = quantiles_levels
        self.pinball_scores = pd.DataFrame()
        self.winkler_scores = pd.DataFrame()

    def compute_pinball_scores(self):
        """
        Utility function to compute the pinball score on the test results
        return: pinball scores computed for each quantile level and each step in the pred horizon
        """
        score = []
        for i, q in enumerate(self.quantiles_levels):
            error = np.subtract(self.y_true, self.pred_quantiles[:, :, i])
            loss_q = np.maximum(q * error, (q - 1) * error)
            score.append(np.expand_dims(loss_q,-1))
        score = np.mean(np.concatenate(score, axis=-1), axis=0)
        self.pinball_scores = pd.DataFrame(score, columns=self.quantiles_levels, index=range(24))
        return score

    def compute_winkler_scores(self):
        """
        Utility function to compute the Winkler's score on the test results
        return: Winkler's scores computed for each quantile level and each step in the pred horizon
        """
        score = []
        for i, q in enumerate(self.quantiles_levels[:len(self.quantiles_levels)//2]):
            l_hat = self.pred_quantiles[:, :, i]
            u_hat = self.pred_quantiles[:, :, -i-1]
            delta = np.subtract(u_hat, l_hat)
            score_i = delta + 2/(1-2*q) * (
                        np.maximum(np.subtract(l_hat, self.y_true), 0) + np.maximum(np.subtract(self.y_true, u_hat), 0))
            score.append(np.expand_dims(score_i, -1))
        score = np.mean(np.concatenate(score, axis=-1), axis=0)
        alpha_levels = [1 - 2 * q for q in self.quantiles_levels[:len(self.quantiles_levels) // 2]]
        self.winkler_scores = pd.DataFrame(score, columns=alpha_levels, index=range(24))
        return score

    def display_scores(self, score_type='pinball', table=True, heatmap=True, summary=True):
        """
        Display the scores in a nicely formatted table.
        score_type: 'pinball' or 'winkler'
        """
        if score_type == 'pinball':
            scores = self.pinball_scores
            x_labels = self.quantiles_levels
        elif score_type == 'winkler':
            scores = self.winkler_scores
            x_labels = [1 - 2 * q for q in self.quantiles_levels[:len(self.quantiles_levels) // 2]]
        else:
            print("Invalid score type. Choose 'pinball' or 'winkler'.")
            return

        # Display the scores in a table
        if table:
            print(f'\n{score_type.capitalize()} Scores:\n')
            print(tabulate(scores, headers='keys', tablefmt='psql'))

        # Display the scores in a heatmap
        if heatmap:
            plt.figure(figsize=(16, 13))
            sns.heatmap(scores, cmap='viridis', annot=True, fmt=".3f", annot_kws={"size": 5})
            plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
            plt.xlabel('Quantile Levels')  # Add x-axis label
            plt.ylabel('Hours')  # Add y-axis label
            plt.title(f'{score_type.capitalize()} Scores')  # Add title
            plt.show()

        if summary:
            print(f'\n{score_type.capitalize()} Summary of scores:\n')
            print(np.mean(scores))

    def plot_scores_3d(self, score_type='pinball'):
        """
        Plot the scores in a 3D graph.
        score_type: 'pinball' or 'winkler'
        """
        if score_type == 'pinball':
            scores = self.pinball_scores
            x_labels = self.quantiles_levels
        elif score_type == 'winkler':
            scores = self.winkler_scores
            x_labels = [1 - 2 * q for q in self.quantiles_levels[:len(self.quantiles_levels) // 2]]
        else:
            print("Invalid score type. Choose 'pinball' or 'winkler'.")
            return

        fig = plt.figure(figsize=(17, 14))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np.arange(scores.shape[1]), np.arange(scores.shape[0]))
        ax.plot_surface(X, Y, scores.values, cmap='coolwarm', alpha=0.7)

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel('Quantile Levels')
        ax.set_ylabel('Hours')
        ax.set_zlabel(f'{score_type.capitalize()} Scores')

        plt.show()

    def export_scores(self, folder_path):
        """
        Export the scores to CSV files.
        folder_path: The path of the folder where the CSV files will be saved.
        """
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Export the pinball_scores DataFrame to a CSV file
        self.pinball_scores.to_csv(os.path.join(folder_path, 'pinball_scores.csv'), index=False)

        # Export the winkler_scores DataFrame to a CSV file
        self.winkler_scores.to_csv(os.path.join(folder_path, 'winkler_scores.csv'), index=False)