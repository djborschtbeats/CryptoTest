import numpy as np
import pandas as pd
import json
import numpy as np

class Model:
    def __init__(self, data, rsi_period=9, ema_short_period=9, ema_long_period=21, vma_period=9, adx_period=14, bollinger_bands_period=50, multiplier=25, rsi_bias=0.25, macd_bias=0.25, vma_bias=0.25, adx_bias=0.25, momentum_bias=0.25, cmf_bias=0.25, fluidity_bias=0.25, alpha=0.5):
        self.data = data
        self.multiplier = multiplier
        
        # Calculate primary indicators
        self.data["RSI"] = self.calculate_rsi(period=int(rsi_period*self.multiplier))
        self.data["EMA_SHORT"] = self.calculate_ema(period=int(ema_short_period*self.multiplier))
        self.data["EMA_LONG"] = self.calculate_ema(period=int(ema_long_period*self.multiplier))
        self.data["MACD"] = self.calculate_macd()
        self.data["VMA"] = self.calculate_vma(period=int(vma_period*self.multiplier))
        self.calculate_bollinger_bands(period=int(bollinger_bands_period*self.multiplier))
        self.calculate_adx(period=int(adx_period*self.multiplier))
        
        # Additional metrics for non-Newtonian fluid model
        self.data["Momentum"] = self.calculate_momentum(period=14*self.multiplier)
        self.data["CMF"] = self.calculate_cmf(period=14*self.multiplier)
        self.data["VWAP"] = self.calculate_vwap()
        self.data["RVI"] = self.calculate_rvi(period=14*self.multiplier)

        # Set biases
        self.rsi_bias = rsi_bias
        self.macd_bias = macd_bias
        self.vma_bias = vma_bias
        self.adx_bias = adx_bias
        self.momentum_bias = momentum_bias
        self.cmf_bias = cmf_bias
        self.fluidity_bias = fluidity_bias
        self.alpha = alpha  # Bias between VWAP and RVI for fluidity

        # Generate the model
        self.data["model"] = self.generate_model()

    def ensure_numeric(self):
        """Ensure relevant columns are numeric."""
        self.data["close"] = pd.to_numeric(self.data["close"], errors='coerce')
        self.data["volume"] = pd.to_numeric(self.data["volume"], errors='coerce')
        self.data["high"] = pd.to_numeric(self.data["high"], errors='coerce')
        self.data["low"] = pd.to_numeric(self.data["low"], errors='coerce')

    def calculate_rsi(self, period=14):
        """Calculates the RSI (Relative Strength Index)."""
        self.ensure_numeric()
        delta = self.data["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, period=20, std_dev_multiplier=2):
        """Calculates Bollinger Bands."""
        self.data['BOL_MID'] = self.data['close'].rolling(window=period).mean()
        self.data['STD_ROLL'] = self.data['close'].rolling(window=period).std()
        self.data['BOL_UP'] = self.data['BOL_MID'] + (std_dev_multiplier * self.data['STD_ROLL'])
        self.data['BOL_LOW'] = self.data['BOL_MID'] - (std_dev_multiplier * self.data['STD_ROLL'])

    def calculate_ema(self, period=9):
        """Calculates the Exponential Moving Average (EMA)."""
        return self.data["close"].ewm(span=period, adjust=False).mean()

    def calculate_macd(self):
        """Calculates the MACD (Moving Average Convergence Divergence)."""
        macd = self.data["EMA_SHORT"] - self.data["EMA_LONG"]
        mean_macd = macd.mean()
        std_macd = macd.std()

        # Clip EMA values to within ±2 sigma
        lower_bound = mean_macd - 2 * std_macd
        upper_bound = mean_macd + 2 * std_macd
        macd_clipped = macd.clip(lower=lower_bound, upper=upper_bound)

        return macd_clipped

    def calculate_vma(self, period=20):
        """Calculates the Volume Moving Average (VMA)."""
        self.ensure_numeric()
        return self.data["volume"].rolling(window=period).mean()

    def calculate_adx(self, period=14):
        """Calculates the Average Directional Index (ADX)."""
        self.ensure_numeric()
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift(1))
        low_close = np.abs(self.data['low'] - self.data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        plus_dm = self.data['high'].diff()
        minus_dm = -self.data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr_smooth = true_range.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()

        self.data['ADX'] = adx
        return adx

    def calculate_momentum(self, period=14):
        """Calculates Momentum as a proxy for Shear Rate."""
        self.ensure_numeric()
        return self.data["close"].diff(period)

    def calculate_cmf(self, period=20):
        """Calculates the Chaikin Money Flow (CMF) as a Flow Index."""
        self.ensure_numeric()
        mf_multiplier = ((self.data['close'] - self.data['low']) - (self.data['high'] - self.data['close'])) / (self.data['high'] - self.data['low'])
        mf_volume = mf_multiplier * self.data['volume']
        cmf = mf_volume.rolling(window=period).sum() / self.data['volume'].rolling(window=period).sum()
        return cmf

    def calculate_vwap(self):
        """Calculates Volume Weighted Average Price (VWAP)."""
        self.ensure_numeric()
        cum_volume = self.data['volume'].cumsum()
        cum_vwap = (self.data['close'] * self.data['volume']).cumsum()
        return cum_vwap / cum_volume
        
    def calculate_rvi(self, period=14):
        """Calculates the Relative Vigor Index (RVI)."""
        self.ensure_numeric()

        # Calculate price change (difference between close prices)
        price_change = self.data['close'].diff()
        
        # Calculate standard deviation of closing prices over the rolling window
        std_dev = self.data['close'].rolling(window=period).std()

        # Calculate the average upward price change
        avg_up = pd.Series(np.where(price_change > 0, std_dev, 0)).rolling(window=period).mean()

        # Calculate the average downward price change
        avg_down = pd.Series(np.where(price_change < 0, std_dev, 0)).rolling(window=period).mean()

        # Calculate the RVI
        rvi = avg_up / (avg_up + avg_down)
        
        return rvi

    def normalize(self, series):
        """Normalizes a series between 0 and 100."""
        return (series - series.min()) / (series.max() - series.min()) * 100

    def generate_model(self):
        """Generates a combined metric."""
        # Normalize the metrics between 0 and 100
        momentum_norm = self.normalize(self.data["Momentum"])
        bollinger_norm = self.normalize(self.data["BOL_MID"])
        cmf_norm = self.normalize(self.data["CMF"])
        vwap_norm = self.normalize(self.data["VWAP"])
        rvi_norm = self.normalize(self.data["RVI"])

        # Combined Fluidity Metric with bias
        fluidity_metric = self.alpha * vwap_norm + (1 - self.alpha) * rvi_norm

        # Combined model with weighted biases
        combined_metric = (self.momentum_bias * momentum_norm +
                           self.cmf_bias * cmf_norm +
                           self.fluidity_bias * fluidity_metric)

        # Normalize the combined metric between 0 and 100
        return self.normalize(combined_metric).clip(0, 100)

    def generate_model_initial(self):
        """Generates a combined metric using RSI, EMA, and VMA."""
        
        self.data["RSI"] = pd.to_numeric(self.data["RSI"], errors='coerce')
        self.data["MACD"] = pd.to_numeric(self.data["MACD"], errors='coerce')
        self.data["VMA"] = pd.to_numeric(self.data["VMA"], errors='coerce')
        self.data["ADX"] = pd.to_numeric(self.data["VMA"], errors='coerce')

        rsi_norm = self.data["RSI"].clip(0, 100)

        # Normalize EMA between 0 and 100
        macd_min = self.data["MACD"].min()
        macd_max = self.data["MACD"].max()
        macd_norm = ((self.data["MACD"] - macd_min) / (macd_max - macd_min)) * 100

        # Normalize VMA between 0 and 100
        vma_min = self.data["VMA"].min()
        vma_max = self.data["VMA"].max()
        vma_norm = ((self.data["VMA"] - vma_min) / (vma_max - vma_min)) * 100

        # Normalize ADX between 0 and 100
        adx_min = self.data["ADX"].min()
        adx_max = self.data["ADX"].max()
        adx_norm = ((self.data["ADX"] - adx_min) / (adx_max - adx_min)) * 100

        combined_metric = self.rsi_bias * rsi_norm + self.macd_bias * macd_norm + self.vma_bias * vma_norm + self.adx_bias * adx_norm

        # Normalize the combined metric between 0 and 100
        combined_metric_min = combined_metric.min()
        combined_metric_max = combined_metric.max()
        combined_metric_norm = ((combined_metric - combined_metric_min) / 
                               (combined_metric_max - combined_metric_min)) * 100

        return combined_metric_norm.clip(0, 100)
