import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlotOptimizationResults:
    def __init__(self, file_path='optimization_results.csv'):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        """Load the CSV file into a pandas DataFrame."""
        return pd.read_csv(self.file_path)

    def plot_results(self):
        # Extract columns for iteration and profit
        iterations = self.data['iteration']
        profits = self.data['profit']

        # Calculate statistics for each iteration
        grouped = self.data.groupby('iteration')['profit']
        min_profits = grouped.min()
        max_profits = grouped.max()
        mean_profits = grouped.mean()

        # Plot the min, max, and mean profits for each iteration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations.unique(), mean_profits, label='Mean Profit', marker='o', linestyle='-', color='blue')
        ax.fill_between(iterations.unique(), min_profits, max_profits, color='gray', alpha=0.3, label='Profit Range (Min-Max)')

        #if x>1, lets do log fit. 
        if len(iterations.unique()) > 1:
            # Add a logarithmic trendline to the mean profits
            x_vals = np.arange(len(mean_profits))
            log_fit = np.polyfit(np.log1p(x_vals), mean_profits, 1)  # Logarithmic fit
            trendline = log_fit[0] * np.log1p(x_vals) + log_fit[1]
            ax.plot(iterations.unique(), trendline, label='Logarithmic Trendline', color='red', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Profit')
        ax.set_title('Optimization Results Over Iterations')
        ax.legend()

        plt.show()

if __name__ == '__main__':
    plotter = PlotOptimizationResults('optimization_results.csv')
    plotter.plot_results()