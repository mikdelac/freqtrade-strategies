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
    
    # Risk thresholds - Note: These thresholds are now in terms of non-annualized volatility
    LOW_VOL_THRESHOLD = 0.01  # Adjusted from 0.15
    MEDIUM_VOL_THRESHOLD = 0.015  # Adjusted from 0.25
    MIN_RISK_THRESHOLD = 0.02  # Adjusted from 0.3
    DEFAULT_RISK_LEVEL = 0.5
    
    # RSI settings
    RSI_PERIOD = 14
    
    # Trading parameters
    can_short: bool = False
    
    # Risk parameters
    har_lags = [1, 5, 22]  # Daily, weekly, monthly lags
    vol_window = IntParameter(10, 30, default=20, space="buy", optimize=True)
    risk_reduction_high = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)
    risk_reduction_medium = DecimalParameter(0.6, 0.9, default=0.8, space="buy", optimize=True)
    
    # Trading parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

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
                "VOL": {
                    "rv_d": {"color": "blue", "type": "line", "title": "Daily RV"},
                    "rv_w": {"color": "green", "type": "line", "title": "Weekly RV"},
                    "rv_m": {"color": "red", "type": "line", "title": "Monthly RV"},
                    "har_vol": {"color": "yellow", "type": "line", "title": "HAR Forecast"},
                },
                "RISK": {
                    "risk_multiplier": {"color": "yellow"},
                },
                "RSI": {
                    "rsi": {"color": "orange"},
                }
            }
        }

    # Risk thresholds mapped to VolatilityRegime
    RISK_MULTIPLIERS = {
        VolatilityRegime.LOW: 1.0,
        VolatilityRegime.MEDIUM: 0.8,  # Will be updated by risk_reduction_medium parameter
        VolatilityRegime.HIGH: 0.5,    # Will be updated by risk_reduction_high parameter
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
        return [("ETH/USDC", "5m")]

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

        # Fit HAR model on realized volatility using proper lags
        if len(dataframe) >= self.MONTHLY_CANDLES:
            try:
                # Get end-of-day values for each RV series
                rv_daily = dataframe.groupby(self._group_by_day(dataframe.index))['rv_d'].last().dropna()
                rv_weekly = dataframe.groupby(self._group_by_day(dataframe.index))['rv_w'].last().dropna()
                rv_monthly = dataframe.groupby(self._group_by_day(dataframe.index))['rv_m'].last().dropna()
                
                # Align the series
                aligned_index = rv_monthly.index
                rv_daily = rv_daily.reindex(aligned_index)
                rv_weekly = rv_weekly.reindex(aligned_index)
                
                # Create HAR model with proper lags
                x = pd.DataFrame({
                    'rv_d': rv_daily,
                    'rv_w': rv_weekly,
                    'rv_m': rv_monthly
                })
                
                # Fit HAR model
                self.har_model = HARX(rv_daily, exog=x[['rv_w', 'rv_m']], lags=self.har_lags)
                self.last_fit = self.har_model.fit(disp='off')
                
                # Generate one-step ahead forecast
                forecasts = self.last_fit.forecast(horizon=1).values
                
                # Map the forecast back to intraday timestamps
                dataframe['har_vol'] = np.nan
                for day in aligned_index:
                    day_mask = (self._group_by_day(dataframe.index) == day)
                    if day in aligned_index:
                        dataframe.loc[day_mask, 'har_vol'] = forecasts[aligned_index.get_loc(day)]
                
                # Forward fill for recent values within the same day only
                dataframe['har_vol'] = dataframe.groupby(self._group_by_day(dataframe.index))['har_vol'].ffill()
                
            except Exception as e:
                print(f"Error fitting HAR model: {e}")
                dataframe['har_vol'] = dataframe['rv_d']
        else:
            dataframe['har_vol'] = dataframe['rv_d']
        
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.RSI_PERIOD)
        
        # Add regime and risk multiplier columns using volatility model
        dataframe[['vol_regime', 'risk_multiplier']] = pd.DataFrame(
            [self.volatility_model.get_regime_and_multiplier(vol) for vol in dataframe['rv_d']],
            index=dataframe.index
        )

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
        
        # Get current risk multiplier based on volatility regime
        risk_multiplier = current_candle['risk_multiplier']
        
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
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['vol_regime'] != VolatilityRegime.HIGH.value) &  # Don't enter in high volatility
            (dataframe['volume'] > 0)
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
            (dataframe['rsi'] > self.sell_rsi.value) |
            (dataframe['vol_regime'] == VolatilityRegime.HIGH.value)  # Exit on high volatility
        )
        
        dataframe.loc[exit_conditions, 'exit_long'] = 1
        return dataframe