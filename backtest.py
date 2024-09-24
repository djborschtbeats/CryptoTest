from model import Model
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

class BacktestStrategy:
    def __init__(self, config):
        """
        Initializes the BacktestStrategy with data and a model class.
        
        :param data: Dataframe with market data
        :param model_class: Class reference for the trading model
        """
        # Create configuration parameters into the model
        self.config = config
        self.apply_model(config)

    def apply_model(self, config=None):
        """
        Applies the configuration to the model.
        
        :param config: A configuration object with model parameters
        """
        # Assign configuration values to model
        self.model = Model(
            config.data,
            rsi_period=config.rsi_period,
            ema_short_period=config.ema_short_period,
            ema_long_period=config.ema_long_period,
            vma_period=config.vma_period,
            bollinger_bands_period=config.bollinger_bands_period,
            multiplier=config.multiplier,
            rsi_bias=config.rsi_bias,
            macd_bias=config.macd_bias,
            vma_bias=config.vma_bias,
        )

    def plot_trades(self, buy_points, sell_points, limit_sell_points, short_points, buy_sell_markers, bank_statement):
        fig = plt.figure(figsize=(10, 9))
        
        # Create gridspec layout: 50% for price and 50% for the 4 indicators
        gs = fig.add_gridspec(6, 1, height_ratios=[5, 1, 1, 1, 1, 1])

        # Price chart in the top 50%
        ax1 = fig.add_subplot(gs[0])

        str_timestamp = self.data["timestamp"]
        ax1.plot(str_timestamp, self.data["close"].astype(float), label="Close Price")
        ax1.plot(str_timestamp, self.data["BOL_UP"].astype(float), label="Upper Bollinger")
        ax1.plot(str_timestamp, self.data["BOL_LOW"].astype(float), label="Lower Bollinger")
        ax1.plot(str_timestamp, self.data["BOL_MID"].astype(float), label="Mid Bollinger")
        
        # Plot Buy points
        for (timestamp, price) in buy_points:
            ax1.plot(timestamp, price, 'gP', label='Buy')

        # Plot Sell points
        for (timestamp, price) in sell_points:
            ax1.plot(timestamp, price, 'rX', label='Sell')

        # Plot Limit Sell points
        for (timestamp, price) in limit_sell_points:
            ax1.plot(timestamp, price, 'yX', label='Sell')

        # Plot Short Sell points
        for (timestamp, price) in short_points:
            ax1.plot(timestamp, price, 'bx', label='Short')

        ax1.set_title("BTC Buy/Sell Points")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USDT)")
        ax1.grid(True)
        
        # Remove repeated legends
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        # RSI in the second subplot
        ax_rsi = fig.add_subplot(gs[1])
        ax_rsi.plot(str_timestamp, self.data["RSI"], label="RSI", color='purple')
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True)

        # EMA in the third subplot
        ax_ema = fig.add_subplot(gs[2])
        ax_ema.plot(str_timestamp, self.data["MACD"], label="MACD", color='green')
        ax_ema.set_ylabel("MACD")
        ax_ema.grid(True)

        # VMA in the fourth subplot
        ax_vma = fig.add_subplot(gs[3])
        ax_vma.plot(str_timestamp, self.data["VMA"], label="VMA", color='orange')
        ax_vma.set_ylabel("VMA")
        ax_vma.grid(True)

        # Metric in the fifth subplot
        ax_metric = fig.add_subplot(gs[4])
        ax_metric.plot(str_timestamp, self.data["model"], label="Model", color='blue')
        ax_metric.plot(str_timestamp, buy_sell_markers, 'ko', label="Buy/Sell Marker", markersize=4)
        # Add a green horizontal line for the Buy threshold
        ax_metric.axhline(y=self.data["THRESHOLD_BUY"].iloc[0], color='green', linestyle='--', label="Buy Threshold")
        # Add a red horizontal line for the Sell threshold
        ax_metric.axhline(y=self.data["THRESHOLD_SELL"].iloc[0], color='red', linestyle='--', label="Sell Threshold")
        ax_metric.set_xlabel("Date")
        ax_metric.set_ylabel("Model")
        ax_metric.grid(True)

        ax_bank = fig.add_subplot(gs[5])
        ax_bank.plot(str_timestamp, bank_statement, label="Bank Statement (Total Value)", color='green')
        ax_bank.set_xlabel("Date")
        ax_bank.set_ylabel("Total Value (Cash + BTC)")
        ax_bank.set_title("Bank Statement Over Time")
        ax_bank.grid(True)

        plt.tight_layout()
        plt.show()
    
    def run_backtest(self, plot_results=True, debug=False):
        """
        Runs a backtest on the strategy with the given configuration.

        :param config: A configuration object with model parameters
        :param sell_threshold: Sell threshold based on model output
        :param buy_threshold: Buy threshold based on model output
        :param initial_money: Initial amount of money to start the strategy
        :param loss_limit_percentage: Maximum loss percentage before selling
        :param enable_shorting: If shorting should be enabled
        :return: The final amount of money after the backtest
        """

        sell_threshold=self.config.sell_threshold
        buy_threshold=self.config.buy_threshold
        initial_money=self.config.initial_money
        loss_limit_percentage=self.config.loss_limit_percentage
        enable_shorting=self.config.enable_shorting
        self.data = self.config.data 
        #Enforce a fee of $100 per trade.
        fee = 100

        money = initial_money
        btc = 0
        last_action = "sell"
        short_position = False  # Track if we are in a short position
        buy_price = 0
        self.data["THRESHOLD_SELL"] = sell_threshold
        self.data["THRESHOLD_BUY"] = buy_threshold

        buy_long_points = []
        sell_points = []
        limit_sell_points = []
        buy_sell_markers = []
        buy_short_points = []

        # Track bank statement (cash + BTC value over time)
        bank_statement = []
        timestamps = []

        for i in range(len(self.data)):
            metric = self.data["model"].iloc[i]
            price = float(self.data["close"].iloc[i])
            timestamp = self.data["timestamp"].iloc[i]

            upper_band = self.data['BOL_UP'].iloc[i]
            lower_band = self.data['BOL_LOW'].iloc[i]

            buy_sell_markers.append(np.nan)
            
            # Calculate current total value (cash + BTC value) and store it
            total_value = money + (btc * price if not short_position else (btc * (buy_price - price) + btc * buy_price))
            bank_statement.append(total_value)
            timestamps.append(timestamp)

            # Limit loss condition: Sell if price drops below the limit loss threshold (for buy_long)
            if (last_action == "buy_long") and (price < buy_price * (1 - loss_limit_percentage / 100)):
                if debug:
                    print(f"Limit loss triggered for buy_long. Price dropped below {loss_limit_percentage}% of {buy_price}")
                money = btc * price - fee
                btc = 0
                last_action = "sell"
                limit_sell_points.append((timestamp, price))
                buy_sell_markers[-1] = 0
                if debug:
                    print(f"Sold at {price} due to limit loss, Cash: {money}")
                continue

            # Limit loss condition for short: Cover if price increases by the limit loss percentage
            if (last_action == "buy_short") and (price > buy_price * (1 + loss_limit_percentage / 100)):
                if debug:
                    print(f"Limit loss triggered for buy_short. Price increased above {loss_limit_percentage}% of {buy_price}")
                money = btc * (buy_price - price) + btc * buy_price - fee # Cover short by buying back BTC
                btc = 0
                buy_price = price  # Track the price where short started
                short_position = False
                last_action = "sell"
                limit_sell_points.append((timestamp, price))
                buy_sell_markers[-1] = 0
                if debug: 
                    print(f"Covered short at {price} due to limit loss, Cash: {money}")
                continue

            # Skip trading if price is outside Bollinger Bands (high volatility)
            if price > upper_band or price < lower_band:
                continue  # Skip trade if the price is outside the bands

            ### Separated Buy Conditions ###
            # Buy Long: Metric is below buy threshold and last action was sell
            if (metric < buy_threshold) and last_action == "sell":
                if debug:
                    print(f"metric: {metric}, last_action: {last_action}")
                btc = money / price  # Buy BTC with all money
                buy_price = price  # Store the buy price for limit loss calculation
                money = 0
                last_action = "buy_long"
                buy_long_points.append((timestamp, price))
                buy_sell_markers[-1] = 1
                if debug:
                    print(f"Bought long at {price}, BTC held: {btc}")
                continue

            # Buy Short: Metric is above sell threshold and shorting is enabled
            if enable_shorting and (metric > sell_threshold) and last_action == "sell" and not short_position:
                if debug:
                    print(f"metric: {metric}, last_action: {last_action}")
                btc = money / price  # Short sell: "borrow" BTC and sell at current price
                buy_price = price  # Store the buy price for limit loss calculation and sell calculations
                money = 0
                last_action = "buy_short"
                short_position = True
                buy_short_points.append((timestamp, price))
                buy_sell_markers[-1] = 0
                if debug:
                    print(f"Shorted at {price}, Cash: {money}")
                continue

            # Sell Long: Metric is above sell threshold and last action was buy_long
            if (metric > sell_threshold) and (last_action == "buy_long"):
                if debug:
                    print(f"metric: {metric}, last_action: {last_action}")
                money = btc * price - fee # Sell BTC for cash
                btc = 0
                last_action = "sell"
                sell_points.append((timestamp, price))
                buy_sell_markers[-1] = 0
                if debug: 
                    print(f"Sold at {price}, Cash: {money}")
                continue

            # Sell Short: Metric is below buy threshold and last action was buy_short
            if (metric < buy_threshold) and (last_action == "buy_short"):
                if debug:
                    print(f"metric: {metric}, last_action: {last_action}")
                money = btc * (buy_price - price) + btc * buy_price - fee # Buy back BTC to cover the short
                btc = 0
                last_action = "sell"
                sell_points.append((timestamp, price))
                buy_sell_markers[-1] = 0
                short_position = False
                if debug:
                    print(f"Covered short at {price}, Cash: {money}")
                continue

        #Close all trades
        # If BTC is still held at the end, sell at the final price
        if btc > 0:
            money = btc * float(self.data["close"].iloc[-1]) - fee

        # If still in a short position at the end, cover the short
        if short_position:
            money -= btc * float(self.data["close"].iloc[-1]) 
            money -= fee
            short_position = False
            if debug:
                print(f"Covered short at final price: {float(self.data['close'].iloc[-1])}, Final cash: {money}")

        if plot_results: 
            self.plot_trades(buy_long_points, sell_points, limit_sell_points, buy_short_points, buy_sell_markers, bank_statement)
        if debug:
            print(f"End Strategy Value: {money}")
        return money