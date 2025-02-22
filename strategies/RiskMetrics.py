# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
from functools import reduce

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from risk_metrics.vix_calculator import VIXCalculator, VIXRegime
from risk_metrics.volatility_models import VolatilityModel, VolatilityRegime


class RiskMetrics(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Risk parameters
    vix_lookback = IntParameter(20, 50, default=30, space="buy", optimize=True)
    vol_window = IntParameter(10, 30, default=20, space="buy", optimize=True)
    risk_reduction_fear = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)
    risk_reduction_neutral = DecimalParameter(0.6, 0.9, default=0.8, space="buy", optimize=True)

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Trading parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Order settings
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "VIX": {
                    "vix": {"color": "red"},
                },
                "RISK": {
                    "risk_multiplier": {"color": "yellow"},
                },
                "RSI": {
                    "rsi": {"color": "orange"},
                }
            }
        }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.vix_calculator = VIXCalculator(lookback_period=self.vix_lookback.value)
        self.volatility_model = VolatilityModel(window_size=self.vol_window.value)

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        if len(dataframe) == 0:
            return dataframe
            
        # Base indicators needed for the strategy
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        
        # Prepare price and volume arrays once
        close_prices = dataframe['close'].values
        high_prices = dataframe['high'].values
        low_prices = dataframe['low'].values
        volumes = dataframe['volume'].values
        
        # Calculate VIX more efficiently
        vix_values = []
        for i in range(len(dataframe)):
            end_idx = i + 1
            start_idx = max(0, end_idx - self.vix_lookback.value)
            data_slice = {
                'close': close_prices[start_idx:end_idx],
                'high': high_prices[start_idx:end_idx],
                'low': low_prices[start_idx:end_idx],
                'volume': volumes[start_idx:end_idx]
            }
            vix_values.append(self.vix_calculator.calculate_vix(data_slice))
        dataframe['vix'] = vix_values
        
        # Calculate rolling volatility
        window = self.vol_window.value
        returns = np.log(close_prices[1:] / close_prices[:-1])
        volatility = []
        
        for i in range(len(dataframe)):
            if i < window:
                # For the first window periods, use available data
                vol = np.std(returns[max(0, i-window):i]) * np.sqrt(252) if i > 0 else 0
            else:
                # For normal calculation, use full window
                vol = np.std(returns[i-window:i]) * np.sqrt(252)
            volatility.append(vol)
            
        dataframe['volatility'] = volatility
        
        # Risk multiplier based on VIX regime
        dataframe['risk_multiplier'] = dataframe['vix'].apply(self.vix_calculator.get_risk_adjustment)
        
        # Combined risk metrics - vectorized calculation
        def get_vol_risk(vol):
            if pd.isna(vol) or not np.isfinite(vol):
                return 0.5  # Default to medium risk for invalid values
            
            if vol <= 15:  # Low volatility threshold
                return 1.0
            elif vol <= 25:  # Medium volatility threshold
                return 0.8
            return 0.5  # High volatility
            
        vol_risks = np.array([get_vol_risk(v) for v in dataframe['volatility']])
        dataframe['combined_risk'] = np.minimum(dataframe['risk_multiplier'].values, vol_risks)
        
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float,
                          min_stake: Optional[float], max_stake: float,
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float,
                          **kwargs) -> Optional[float]:
        """Adjust position size based on risk metrics"""
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # Get current risk multiplier
        risk_multiplier = current_candle['combined_risk']
        
        # Adjust position size
        return max_stake * risk_multiplier

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[:, 'enter_long'] = 0
        
        entry_conditions = (
            (dataframe['rsi'] < self.buy_rsi.value) &  # Oversold
            (dataframe['combined_risk'] > 0.5) &  # Acceptable risk level
            (dataframe['volume'] > 0)  # Ensure volume exists
        )
        
        dataframe.loc[entry_conditions, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[:, 'exit_long'] = 0
        
        exit_conditions = (
            (dataframe['rsi'] > self.sell_rsi.value) |  # Overbought
            (dataframe['combined_risk'] < 0.3)  # High risk environment
        )
        
        dataframe.loc[exit_conditions, 'exit_long'] = 1
        return dataframe