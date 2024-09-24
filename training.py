import random
import csv
from model import Model  # Import the external Model class
from backtest import BacktestStrategy
from plot_results import PlotOptimizationResults
import time 
import pickle
import os

class Config:
    def __init__(self, data, initial_money=100000, sell_threshold=70, buy_threshold=15, loss_limit_percentage=5, enable_shorting=False, rsi_period=9, ema_short_period=9, ema_long_period=21, vma_period=9, adx_period=14, bollinger_bands_period=9, multiplier=50, rsi_bias=0.25, macd_bias=0.25, vma_bias=0.25, adx_bias=0.25):
        self.data = data
        self.initial_money = initial_money
        self.loss_limit_percentage = loss_limit_percentage
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.enable_shorting = enable_shorting
        self.rsi_period = rsi_period
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.vma_period = vma_period
        self.adx_period = adx_period
        self.bollinger_bands_period = bollinger_bands_period
        self.multiplier = multiplier
        self.rsi_bias = rsi_bias 
        self.macd_bias = macd_bias
        self.vma_bias = vma_bias
        self.adx_bias = adx_bias

    def save(self, filename):
        """
        Save the given config object to a file.
        
        Args:
            filename (str): The name of the file to save the config to.
        """
        # Get the directory from the filename
        directory = os.path.dirname(filename)
        
        # Check if the directory exists, if not create it
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

        # Save the file
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(self, file)
        print(f"Configuration saved as {filename}.pkl")
        return True


    @classmethod
    def load(cls, filename):
        """
        Load a config object from a file and return a new instance.
        
        Args:
            filename (str): The name of the file to load the config from (without .pkl extension).
        
        Returns:
            Config: The loaded configuration object.
        """
        with open(f"{filename}.pkl", "rb") as file:
            loaded_config = pickle.load(file)
        print(f"Configuration loaded from {filename}.pkl")
        return loaded_config

class TrainingModel:
    def __init__(self, data):
        self.data = data
        
    def backtest_strategy(self, config, plot_results=False):
        bts = BacktestStrategy(config)
        profit = bts.run_backtest(plot_results=plot_results)
        return profit

    def generate_random_config(self):
        """Generate a random configuration with specified ranges for each parameter."""

        return Config(
            data=self.data,
            initial_money=100000,  # Random initial money between 50,000 and 200,000
            sell_threshold=random.uniform(60, 90),  # Random sell threshold between 50 and 100
            buy_threshold=random.uniform(10, 40),  # Random buy threshold between 5 and 50
            loss_limit_percentage=random.uniform(5, 15),  # Random loss limit percentage between 1% and 10%
            enable_shorting=False, #random.choice([True, False]),  # Randomly enable or disable shorting
            rsi_period=9, #random.randint(5, 20),  # Random RSI period between 5 and 20
            ema_short_period=9, #random.randint(5, 20),  # Random EMA short period between 5 and 20
            ema_long_period=21, #random.randint(20, 50),  # Random EMA long period between 20 and 50
            vma_period=9, #random.randint(5, 20),  # Random VMA period between 5 and 20
            adx_period=9,
            bollinger_bands_period=9, #random.randint(10, 100),  # Random Bollinger Bands period between 10 and 100
            multiplier=random.randint(1, 30),  # Random multiplier between 10 and 100
            rsi_bias=random.uniform(-1, 1),
            macd_bias=random.uniform(-1, 1),
            vma_bias=random.uniform(-1, 1),
            adx_bias=random.uniform(-1, 1),
        )

    def run_genetic_algorithm(self, iterations=50, population_size=20, mutation_rate=0.2, mutation_bias=0.1, output_csv='optimization_results.csv', plot_results=False, debug=False):
        # Initialize population with random configurations
        population = [self.generate_random_config() for _ in range(population_size)]
        results = []
        # Initialize the overall best configuration and profit
        best_overall_config = None
        best_overall_profit = -float('inf')

        # Write results to CSV
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow([
                'sell_threshold', 'buy_threshold', 'loss_limit_percentage', 'enable_shorting', 'rsi_period',
                'ema_short_period', 'ema_long_period', 'vma_period', 'adx_period', 'bollinger_bands_period', 'multiplier', 'rsi_bias', 'macd_bias', 'vma_bias', 'adx_bias', 'profit', 'iteration'
            ])

        for iteration in range(iterations):
            print(f"")
            print(f"Iteration {iteration + 1}/{iterations}")
            # Start the timer
            start_time = time.time()

            # Evaluate each configuration in the population
            profits = []
            for config in population:
                profit = self.backtest_strategy(config, plot_results=plot_results)
                profits.append((config, profit))

                #log the datas
                config_log_data = [
                    config.sell_threshold, config.buy_threshold, config.loss_limit_percentage,
                    config.enable_shorting, config.rsi_period, config.ema_short_period,
                    config.ema_long_period, config.vma_period, config.adx_period, config.bollinger_bands_period,
                    config.multiplier, config.rsi_bias, config.macd_bias, config.vma_bias, config.adx_bias,
                    profit, iteration
                ]
                with open(output_csv, mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(config_log_data)
            print(f"{profits}") 

            # Sort by profit (descending order)
            profits.sort(key=lambda x: x[1], reverse=True)
            
            # Save the best configuration of this iteration to results
            best_config, best_profit = profits[0]
            # Update overall best configuration and profit if the current best is better
            if best_profit > best_overall_profit:
                best_overall_config = best_config
                best_overall_profit = best_profit

            log_data = [
                best_config.sell_threshold, best_config.buy_threshold, best_config.loss_limit_percentage,
                best_config.enable_shorting, best_config.rsi_period, best_config.ema_short_period,
                best_config.ema_long_period, best_config.vma_period, best_config.adx_period, best_config.bollinger_bands_period,
                best_config.multiplier, best_config.rsi_bias, best_config.macd_bias, best_config.vma_bias,
                best_config.adx_bias,
                best_profit, iteration
            ]
            '''
            with open(output_csv, mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log_data)
            '''
            results.append(log_data)


            # Select the top configurations for breeding
            top_configs = [config for config, _ in profits[:population_size // 2]]

            # Generate new population through crossover and mutation
            new_population = []

            while len(new_population) < population_size:
                parent1, parent2 = random.sample(top_configs, 2)

                # Crossover: create a new Config by combining attributes of parents
                child = Config(
                    self.data,
                    sell_threshold=random.choice([parent1.sell_threshold, parent2.sell_threshold]),
                    buy_threshold=random.choice([parent1.buy_threshold, parent2.buy_threshold]),
                    loss_limit_percentage=random.choice([parent1.loss_limit_percentage, parent2.loss_limit_percentage]),
                    enable_shorting=random.choice([parent1.enable_shorting, parent2.enable_shorting]),
                    rsi_period=random.choice([parent1.rsi_period, parent2.rsi_period]),
                    ema_short_period=random.choice([parent1.ema_short_period, parent2.ema_short_period]),
                    ema_long_period=random.choice([parent1.ema_long_period, parent2.ema_long_period]),
                    vma_period=random.choice([parent1.vma_period, parent2.vma_period]),
                    bollinger_bands_period=random.choice([parent1.bollinger_bands_period, parent2.bollinger_bands_period]),
                    multiplier=random.choice([parent1.multiplier, parent2.multiplier]),
                    rsi_bias=random.choice([parent1.rsi_bias, parent2.rsi_bias]),
                    vma_bias=random.choice([parent1.vma_bias, parent2.vma_bias]),
                    macd_bias=random.choice([parent1.macd_bias, parent2.macd_bias]),
                    adx_bias=random.choice([parent1.adx_bias, parent2.adx_bias]),
                )

                # Mutation: randomly alter some of the child's attributes
                if random.random() < mutation_rate:
                    child.sell_threshold = float(child.sell_threshold * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                if random.random() < mutation_rate:
                    child.buy_threshold = float(child.buy_threshold * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                if random.random() < mutation_rate:
                    child.loss_limit_percentage = float(child.loss_limit_percentage * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                #if random.random() < mutation_rate:
                #    child.enable_shorting = random.choice([True, False])
                #if random.random() < mutation_rate:
                #    child.rsi_period = random.randint(5, 20)
                #if random.random() < mutation_rate:
                #    child.ema_short_period = random.randint(5, 20)
                #if random.random() < mutation_rate:
                #    child.ema_long_period = random.randint(20, 50)
                #if random.random() < mutation_rate:
                #    child.vma_period = random.randint(5, 20)
                #if random.random() < mutation_rate:
                #    child.bollinger_bands_period = random.randint(10, 100)
                if random.random() < mutation_rate:
                    child.multiplier = max(1, int(child.multiplier * (1.0 + random.uniform(-mutation_bias, mutation_bias))))

                if random.random() < mutation_rate:
                    child.rsi_bias = float(child.rsi_bias * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                if random.random() < mutation_rate:
                    child.macd_bias = float(child.macd_bias * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                if random.random() < mutation_rate:
                    child.vma_bias = float(child.vma_bias * (1.0 + random.uniform(-mutation_bias, mutation_bias)))
                if random.random() < mutation_rate:
                    child.adx_bias = float(child.adx_bias * (1.0 + random.uniform(-mutation_bias, mutation_bias)))

                new_population.append(child)

            # Replace old population with the new one
            population = new_population

            # End the timer and calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Iteration {iteration + 1} completed in {elapsed_time:.2f} seconds")

            # Estimate the remaining time
            iterations_left = iterations - (iteration + 1)
            estimated_remaining_time = iterations_left * elapsed_time

            # Convert estimated time to minutes and seconds
            minutes, seconds = divmod(estimated_remaining_time, 60)
            if minutes > 0:
                print(f"Estimated time remaining: {int(minutes)} minutes and {seconds:.2f} seconds")
            else:
                print(f"Estimated time remaining: {seconds:.2f} seconds")

            if debug:                    
                por = PlotOptimizationResults(file_path=output_csv)
                por.plot_results()
                
        if debug is False:      
            por = PlotOptimizationResults(file_path=output_csv)
            por.plot_results()

        # Return the best overall configuration
        return best_overall_config