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
                "trend_high": {"color": "green", "type": "line", "style": "dotted"},
                "trend_low": {"color": "red", "type": "line", "style": "dotted"},
                "trend_close": {"color": "blue", "type": "line", "style": "dotted"}
            },
            "subplots": {
                "VOL": {
                    "rv_d": {"color": "blue", "type": "line", "title": "Daily RV"},
                    "rv_w": {"color": "green", "type": "line", "title": "Weekly RV"},
                    "rv_m": {"color": "red", "type": "line", "title": "Monthly RV"},
                    "rv_1h": {"color": "purple", "type": "line", "title": "1-Hour RV"},
                    "rv_1h_change": {"color": "cyan", "type": "line", "title": "1h RV Change"}
                },
                "RISK": {
                    "risk_multiplier": {"color": "yellow"},
                    "rising_vol_1h": {"color": "magenta", "type": "line", "title": "Rising Vol"}
                },
                "RSI": {
                    "rsi": {"color": "orange"},
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
            min_points=10,  # Require more points for a valid trendline
            min_slope=0.0001,
            min_strength=0.3,  # Lower strength requirement for visualization
            angle_threshold=90  # Allow steeper angles
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

        # Calculate trendlines
        lookback = 200  # Use last 20 candles for trendline calculation
        
        # Get current trendlines for high, low, and close prices
        high_trendline = self.trend_analyzer.get_current_trendline(dataframe, lookback, 'high')
        low_trendline = self.trend_analyzer.get_current_trendline(dataframe, lookback, 'low')
        close_trendline = self.trend_analyzer.get_current_trendline(dataframe, lookback, 'close')
        
        # Initialize trendline columns with NaN
        dataframe['trend_high'] = np.nan
        dataframe['trend_low'] = np.nan
        dataframe['trend_close'] = np.nan
        
        # Calculate trendline values if they exist
        if high_trendline:
            indices = np.arange(high_trendline.start_index, high_trendline.end_index + 1)
            normalized_indices = indices - indices[0]
            dataframe.loc[indices, 'trend_high'] = (
                high_trendline.slope * normalized_indices + high_trendline.intercept
            )
            
        if low_trendline:
            indices = np.arange(low_trendline.start_index, low_trendline.end_index + 1)
            normalized_indices = indices - indices[0]
            dataframe.loc[indices, 'trend_low'] = (
                low_trendline.slope * normalized_indices + low_trendline.intercept
            )
            
        if close_trendline:
            indices = np.arange(close_trendline.start_index, close_trendline.end_index + 1)
            normalized_indices = indices - indices[0]
            dataframe.loc[indices, 'trend_close'] = (
                close_trendline.slope * normalized_indices + close_trendline.intercept
            )

        # Calculate 5-minute returns
        dataframe['returns'] = np.log(dataframe['close'] / dataframe['close'].shift(1))
        
        # Calculate realized volatility for different frequencies
        dataframe['rv_d'] = self._calculate_realized_volatility(
            dataframe['returns'],
            window=self.DAILY_CANDLES
        )

        dataframe['rv_w'] = self._calculate_realized_volatility(
            dataframe['returns'],
            window=self.WEEKLY_CANDLES
        )

        dataframe['rv_m'] = self._calculate_realized_volatility(
            dataframe['returns'],
            window=self.MONTHLY_CANDLES
        )

        # Calculate realized volatility for 1-hour period
        dataframe['rv_1h'] = self._calculate_realized_volatility(
            dataframe['returns'],
            window=self.ONE_HOUR_CANDLES
        )
        
        # Calculate volatility change
        dataframe['rv_d_change'] = dataframe['rv_d'].pct_change()
        dataframe['rv_w_change'] = dataframe['rv_w'].pct_change()
        dataframe['rv_m_change'] = dataframe['rv_m'].pct_change()
        dataframe['rv_1h_change'] = dataframe['rv_1h'].pct_change()
        
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.RSI_PERIOD)
        
        # RSI crossing signals
        dataframe['rsi_cross_30'] = (
            (dataframe['rsi'] > 30) & 
            (dataframe['rsi'].shift(1) <= 30)
        )
        
        # Add regime and risk multiplier columns using volatility model
        dataframe[['vol_regime', 'risk_multiplier']] = pd.DataFrame(
            [self.volatility_model.get_regime_and_multiplier(vol) for vol in dataframe['rv_d']],
            index=dataframe.index
        )
        
        # Mark periods of high volatility based on 1-hour realized volatility
        dataframe['high_vol_1h'] = (dataframe['rv_1h'] > self.high_vol_threshold_1h.value).astype(int)
        
        # Mark periods of rising volatility based on 1-hour change threshold
        dataframe['rising_vol_1h'] = (dataframe['rv_1h_change'] > self.rv_1h_change_threshold.value).astype(int)

        print(dataframe["rsi"])
        print(dataframe["trend_high"])
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
        Entry Conditions:
        1. RSI crosses above 30
        2. Volatility is rising (positive change)
        3. Current volatility is above medium threshold
        """
        dataframe.loc[:, 'enter_long'] = 0
        
        entry_conditions = (
            dataframe['rsi_cross_30'] &  # RSI crosses above 30
            (dataframe['rv_1h_change'] > self.rv_1h_change_threshold.value) &  # Rising volatility
            (dataframe['rv_d'] > self.MEDIUM_VOL_THRESHOLD) &  # Above medium threshold
            (dataframe['volume'] > 0)  # Ensure volume
        )
        
        dataframe.loc[entry_conditions, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No exit signals since we're using ROI-based exits
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe