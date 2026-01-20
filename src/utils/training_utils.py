import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def plot_training_history(histories, model_name, save_dir):
    """
    Plots the mean training/validation curves for metrics.
    histories: list of history objects (from multiple runs) or single history.
    """
    if not isinstance(histories, list):
        histories = [histories]

    metrics = ['accuracy', 'recall', 'f1_score', 'loss']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        if metric not in histories[0].history:
            continue
            
        plt.subplot(2, 2, i+1)
        
        # Aggregate data from multiple runs
        train_values = []
        val_values = []
        
        for h in histories:
            train_values.append(h.history[metric])
            val_values.append(h.history[f'val_{metric}'])
            
        train_mean = np.mean(train_values, axis=0)
        val_mean = np.mean(val_values, axis=0)
        
        epochs = range(1, len(train_mean) + 1)
        
        plt.plot(epochs, train_mean, 'b-', label=f'Training {metric}')
        plt.plot(epochs, val_mean, 'r-', label=f'Validation {metric}')
        
        if len(histories) > 1:
            train_std = np.std(train_values, axis=0)
            val_std = np.std(val_values, axis=0)
            plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='b')
            plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='r')
            
        plt.title(f'{model_name} - {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_history_plot.png'))
    plt.close()

def plot_comparison_boxplots(results_df, save_dir):
    """
    Plots boxplots comparing models.
    results_df: DataFrame with columns ['Model', 'Metric', 'Value', 'Set'] (Set=Train/Test)
    """
    metrics = results_df['Metric'].unique()
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        subset = results_df[results_df['Metric'] == metric]
        
        sns.boxplot(x='Model', y='Value', hue='Set', data=subset)
        plt.title(f'Comparison of {metric}')
        plt.grid(True, axis='y')
        
        plt.savefig(os.path.join(save_dir, f'boxplot_{metric}.png'))
        plt.close()
