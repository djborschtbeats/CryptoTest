import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from training import TrainingModel, Config
from backtest import BacktestStrategy

class BTCDataReader:
    def __init__(self, folder="btc_data"):
        self.folder = folder
        self.data = pd.DataFrame()

    def load_data(self, from_date_str, to_date_str):
        from_date = datetime.strptime(from_date_str, "%Y-%m-%d")
        to_date = datetime.strptime(to_date_str, "%Y-%m-%d")

        for file_name in sorted(os.listdir(self.folder)):
            if file_name.startswith("btc_data_"):
                date_str = file_name.replace("btc_data_", "").replace(".json", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if from_date <= file_date <= to_date:
                    file_path = os.path.join(self.folder, file_name)
                    self._read_file(file_path)

    def _read_file(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            self.data = pd.concat([self.data, df], ignore_index=True)
    
    def train_model(self, debug=False): 
        tm = TrainingModel(self.data)
        return tm.run_genetic_algorithm(iterations=10, population_size=10, mutation_rate=0.25, mutation_bias=0.25, output_csv='optimization_results.csv', debug=False)

    def plot_data(self):
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit='ms')
        plt.figure(figsize=(10, 6))
        plt.plot(self.data["timestamp"], self.data["close"].astype(float), label="Close Price")
        plt.title("BTC Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USDT)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_metric(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data["timestamp"], self.data["model"], label="Buy/Sell Metric")
        plt.title("BTC Buy/Sell Model")
        plt.xlabel("Date")
        plt.ylabel("Model (0-100)")
        plt.grid(True)
        plt.legend()
        plt.show()

# Usage
# Current Test
#reader.load_data(from_date_str="2024-09-01", to_date_str="2024-09-13")
# Backtest
#reader.load_data(from_date_str="2024-05-01", to_date_str="2024-09-13")
# Retest 
#reader.load_data(from_date_str="2023-09-16", to_date_str="2024-01-1")
#

#print(f"Final amount after backtest: ${final_amount:.2f}")

def train(name="test"):
    reader = BTCDataReader()
    reader.load_data(from_date_str="2024-07-01", to_date_str="2024-09-20")
    config = reader.train_model()
    config.save("configs/"+name)
    return config 

def test(config):
    # Show the "winner on the training set"
    reader0 = BTCDataReader()
    reader0.load_data(from_date_str="2024-07-01", to_date_str="2024-09-20")
    config.data = reader0.data
    bt = BacktestStrategy(config)
    bt.run_backtest()

    # Show the "winner on other data set"
    reader1 = BTCDataReader()
    reader1.load_data(from_date_str="2024-01-01", to_date_str="2024-05-01")
    config.data = reader1.data
    bt = BacktestStrategy(config)
    bt.run_backtest()

    # Show it on the most recent timeline
    reader2 = BTCDataReader()
    reader2.load_data(from_date_str="2024-09-01", to_date_str="2024-09-20")
    config.data = reader2.data
    bt = BacktestStrategy(config)
    bt.run_backtest()

config = train()
test(config)