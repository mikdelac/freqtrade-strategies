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
from arch.univariate import HARX
from risk_metrics.volatility_models import VolatilityModel, VolatilityRegime
from trend_analysis.trendlines import TrendAnalysis


class RiskMetrics(IStrategy):
    """
    RiskMetrics strategy using HAR-RV (Heterogeneous Autoregression Realized Volatility) model
    for volatility forecasting and risk management.
    """
    INTERFACE_VERSION = 3

    # Timeframe settings
    timeframe = "5m"
    MINUTES_IN_DAY = 24 * 60
    MINUTES_PER_CANDLE = 5
    CANDLES_PER_DAY = MINUTES_IN_DAY // MINUTES_PER_CANDLE  # 288 5-min candles per day
    TRADING_DAYS_PER_YEAR = 252
    WEEKS_PER_MONTH = 4.33
    TRADING_DAYS_PER_WEEK = 5
    TRADING_DAYS_PER_MONTH = 22
    HOURS_PER_DAY = 24
    WEEKS_PER_YEAR = 52
    MONTHS_PER_YEAR = 12
    
    # Volatility calculation constants
    DAILY_CANDLES = CANDLES_PER_DAY  # Target: 288 candles
    WEEKLY_CANDLES = CANDLES_PER_DAY * TRADING_DAYS_PER_WEEK  # Target: 1440 candles
    MONTHLY_CANDLES = CANDLES_PER_DAY * TRADING_DAYS_PER_MONTH  # Target: 6336 candles
    ONE_HOUR_CANDLES = 12  # 12 candles = 60 minutes
    
    # Risk thresholds - Note: These thresholds are now in terms of non-annualized volatility
    LOW_VOL_THRESHOLD = 0.01
    MEDIUM_VOL_THRESHOLD = 0.015
    
    # RSI settings
    RSI_PERIOD = 14
    
    # Trading parameters
    can_short: bool = False
    
    # Risk parameters
    risk_reduction_high = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)
    risk_reduction_medium = DecimalParameter(0.6, 0.9, default=0.8, space="buy", optimize=True)
    high_vol_threshold_1h = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    rv_1h_change_threshold = DecimalParameter(0.05, 0.10, default=0.01, space="buy", optimize=True)

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "360": 0.15,  # Exit after 6 hours if profit is 15%
        "240": 0.10,  # Exit after 4 hours if profit is 10%
        "120": 0.07,  # Exit after 2 hours if profit is 7%
        "60": 0.05,   # Exit after 1 hour if profit is 5%
        "30": 0.03,   # Exit after 30 min if profit is 3%
        "0": 0.02     # Exit immediately if profit is 2%
    }

    # Disable stoploss since we're using ROI-based exits only
    stoploss = -0.1  # Effectively disabled
    trailing_stop = False
    use_exit_signal = False  # Disable exit signals since we're using ROI
    exit_profit_only = False  # Only exit in profit
    ignore_roi_if_entry_signal = False  # Don't ignore ROI even if we have a new entry signal

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Order settings
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter"},
                "all_lows": {"color": "green", "type": "scatter"},
                "resistance_line": {"color": "red", "width": 2.0},
                "support_line": {"color": "green", "width": 2.0},
                "current_trendline": {"color": "purple", "width": 2.0},
                "trendline_forecast": {"color": "purple", "style": "dashdot", "width": 1.5}
            },
            "subplots": {
                "ATR": {
                    "atr": {"color": "blue"}
                }
            }
        }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Initialize volatility model with thresholds and risk multipliers
        self.volatility_model = VolatilityModel(
            low_threshold=self.LOW_VOL_THRESHOLD,
            medium_threshold=self.MEDIUM_VOL_THRESHOLD,
            risk_multipliers={
                VolatilityRegime.LOW: 1.0,
                VolatilityRegime.MEDIUM: self.risk_reduction_medium.value,
                VolatilityRegime.HIGH: self.risk_reduction_high.value
            }
        )
        self.trend_analyzer = TrendAnalysis(
            min_points=2,  # Reduced minimum points
            min_slope=0.00001,  # Reduced minimum slope
            min_strength=0.2,  # Reduced strength requirement
            angle_threshold=85  # Increased angle threshold
        )
        self.har_model = None
        self.last_fit = None

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
        return []  # No informative pairs needed

    def _group_by_day(self, timestamps: pd.Series) -> pd.Series:
        """
        Group timestamps by trading day
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Series with day grouping
        """
        return pd.to_datetime(timestamps).dt.date

    # Define scaling factors for different frequencies
    SCALING_FACTORS = {
        'daily': TRADING_DAYS_PER_YEAR,
        'weekly': WEEKS_PER_YEAR,
        'monthly': MONTHS_PER_YEAR
    }

    def _calculate_realized_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculate realized volatility using sum of squared returns.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Realized volatility series (non-annualized)
        """
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Sum over the window to get realized variance
        realized_var = squared_returns.rolling(window=window).sum()
        
        # Take square root to get realized volatility
        return np.sqrt(realized_var)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame, including ATR.
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Calculate ATR and store it for visualization
        dataframe['atr'] = self.volatility_model.calculate_atr(dataframe)

        # Initialize marker columns for support and resistance points
        dataframe['all_highs'] = np.nan
        dataframe['all_lows'] = np.nan
        dataframe['resistance_line'] = np.nan
        dataframe['support_line'] = np.nan
        dataframe['current_trendline'] = np.nan
        dataframe['trendline_forecast'] = np.nan
        
        # Calculate trendlines using local maxima/minima
        lookback = 2000  # Use last 300 candles for trendline calculation

        # Don't calculate trendlines if we don't have enough data
        if len(dataframe) < 30:
            return dataframe
            
        recent_data = dataframe.tail(lookback).copy()
        
        # Find swing points for both highs and lows with ATR-based filtering
        high_swing_points = self.trend_analyzer._find_swing_points(
            recent_data['high'].values, 
            price_type='high', 
            min_points=2,
            distance=5
        )
        
        low_swing_points = self.trend_analyzer._find_swing_points(
            recent_data['low'].values, 
            price_type='low', 
            min_points=2,
            distance=5
        )
        
        # Mark swing points on the chart
        for idx, price in high_swing_points:
            actual_idx = len(dataframe) - lookback + idx
            dataframe.loc[actual_idx, 'all_highs'] = price
            
        for idx, price in low_swing_points:
            actual_idx = len(dataframe) - lookback + idx
            dataframe.loc[actual_idx, 'all_lows'] = price
            
        # Find resistance (high) trendlines
        resistance_lines = self.trend_analyzer.find_trendlines(
            recent_data,
            high_swing_points,
            price_type='high',
            min_points=2
        )
        print("resistance_lines: ", resistance_lines)
        
        # Find support (low) trendlines
        support_lines = self.trend_analyzer.find_trendlines(
            recent_data, 
            low_swing_points,
            price_type='low',
            min_points=2
        )
        
        # Plot the strongest resistance line (if any)
        if resistance_lines and len(resistance_lines) > 0:
            resistance = max(resistance_lines, key=lambda t: t.strength)
            start_idx = len(dataframe) - lookback + resistance.start_index
            end_idx = len(dataframe) - lookback + resistance.end_index
            
            # Plot resistance line
            indices = np.arange(start_idx, end_idx + 1)
            normalized_indices = np.arange(len(indices))
            dataframe.loc[indices, 'resistance_line'] = (
                resistance.slope * normalized_indices + resistance.intercept
            )
            
            # Extrapolate resistance line into the future
            steps_forward = 10
            future_values = self.trend_analyzer.extrapolate_trendline(resistance, steps_forward)
            future_indices = np.arange(
                end_idx + 1,
                end_idx + steps_forward + 1
            )
            
            for i, idx in enumerate(future_indices):
                if idx < len(dataframe):
                    dataframe.loc[idx, 'resistance_line'] = future_values[i]
        
        # Plot the strongest support line (if any)
        if support_lines and len(support_lines) > 0:
            support = max(support_lines, key=lambda t: t.strength)
            start_idx = len(dataframe) - lookback + support.start_index
            end_idx = len(dataframe) - lookback + support.end_index
            
            # Plot support line
            indices = np.arange(start_idx, end_idx + 1)
            normalized_indices = np.arange(len(indices))
            dataframe.loc[indices, 'support_line'] = (
                support.slope * normalized_indices + support.intercept
            )
            
            # Extrapolate support line into the future
            steps_forward = 10
            future_values = self.trend_analyzer.extrapolate_trendline(support, steps_forward)
            future_indices = np.arange(
                end_idx + 1,
                end_idx + steps_forward + 1
            )
            
            for i, idx in enumerate(future_indices):
                if idx < len(dataframe):
                    dataframe.loc[idx, 'support_line'] = future_values[i]

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float,
                          min_stake: Optional[float], max_stake: float,
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float,
                          **kwargs) -> Optional[float]:
        """Adjust position size based on volatility regime"""
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        return max_stake * current_candle['risk_multiplier']

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal is always 0 since we're just visualizing points
        """
        dataframe.loc[:, 'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal is always 0 since we're just visualizing points
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe