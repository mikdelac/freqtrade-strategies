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
# Import the trendline functions
from strategies.trend_analysis.trendline import gentrends, segtrends, rank_trendlines


class RiskMetrics(IStrategy):
    """
    RiskMetrics strategy using HAR-RV (Heterogeneous Autoregression Realized Volatility) model
    for volatility forecasting and risk management.
    
    Features:
    - Volatility forecasting using HAR-RV model
    - Risk-adjusted position sizing based on volatility regime
    - Trendline analysis with support and resistance identification
    - Trendline ranking based on price proximity and touch frequency
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
    
    # Trendline parameters
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.002, space="buy", optimize=True)
    trendline_touch_weight = DecimalParameter(1.5, 3.0, default=2.5, space="buy", optimize=True)

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
        # Basic configuration with default plots
        plot_config = {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter"},
                "all_lows": {"color": "green", "type": "scatter"},
                "Max Line": {"color": "red", "width": 2.0},
                "Min Line": {"color": "green", "width": 2.0},
                "current_trendline": {"color": "purple", "width": 2.0},
                "trendline_forecast": {"color": "purple", "style": "dashdot", "width": 1.5}
            },
            "subplots": {
                "ATR": {
                    "atr": {"color": "blue"}
                }
            }
        }
        
        # Define the maximum number of segments we'll use
        max_segments = 10
        
        # Add individual trend lines from segtrends for each segment
        for i in range(max_segments):
            # Max lines (resistance)
            line_name = f'Max_Line_{i}'
            intensity = max(30, 100 - i * 5)  # Decreasing intensity for weaker lines
            plot_config["main_plot"][line_name] = {
                "color": f"rgb(255, {intensity}, {intensity})",
                "width": max(0.5, 2.0 - i * 0.1)  # Thinner lines for weaker resistance
            }
            
            # Min lines (support)
            line_name = f'Min_Line_{i}'
            intensity = max(30, 100 - i * 5)  # Decreasing intensity for weaker lines
            plot_config["main_plot"][line_name] = {
                "color": f"rgb({intensity}, 255, {intensity})",
                "width": max(0.5, 2.0 - i * 0.1)  # Thinner lines for weaker support
            }
        
        # Add ranked trendlines with distinct colors and styles
        # Top 3 ranked maxlines (resistance)
        for i in range(1, 4):
            line_name = f'Ranked_Max_{i}'
            plot_config["main_plot"][line_name] = {
                "color": f"rgb(220, 50, {50 + i * 50})",  # Distinct red shades
                "width": 3.0 - (i - 1) * 0.5,  # Thicker lines for higher ranks
                "style": "solid" if i == 1 else ("dashdot" if i == 2 else "dotted")
            }
            
        # Top 3 ranked minlines (support)
        for i in range(1, 4):
            line_name = f'Ranked_Min_{i}'
            plot_config["main_plot"][line_name] = {
                "color": f"rgb(50, 220, {50 + i * 50})",  # Distinct green shades
                "width": 3.0 - (i - 1) * 0.5,  # Thicker lines for higher ranks
                "style": "solid" if i == 1 else ("dashdot" if i == 2 else "dotted")
            }
            
        return plot_config
    
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
            angle_threshold=90  # Increased angle threshold
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
        
        # Initialize columns for main trend lines
        dataframe['Max Line'] = np.nan
        dataframe['Min Line'] = np.nan
        
        # Define the maximum number of segments we'll use
        max_segments = 10
        
        # Initialize columns for all individual segment trend lines
        for i in range(max_segments):
            dataframe[f'Max_Line_{i}'] = np.nan
            dataframe[f'Min_Line_{i}'] = np.nan
        
        # Calculate trendlines using local maxima/minima
        lookback = 200  # Use last 2000 candles for trendline calculation

        # Don't calculate trendlines if we don't have enough data
        if len(dataframe) < 30:
            return dataframe
            
        recent_data = dataframe.tail(lookback).copy()
        
        # Find global trendlines using gentrends
        try:
            # Generate trends using the gentrends function
            # This gives us the main max and min lines (resistance and support)
            trends = gentrends(recent_data, field='close', window=1/3.0)
            
            # Map the trend lines to the dataframe
            last_idx = len(dataframe) - len(recent_data)
            for i in range(len(trends)):
                current_idx = last_idx + i
                if current_idx < len(dataframe):
                    dataframe.loc[current_idx, 'Max Line'] = trends['Max Line'].iloc[i]
                    dataframe.loc[current_idx, 'Min Line'] = trends['Min Line'].iloc[i]
        except Exception as e:
            # Fail gracefully if gentrends fails
            print(f"Error in gentrends: {e}")
        
        # Use segtrends with different segment counts to find multiple resistance and support lines
        # Try with max_segments + 1 segments to get max_segments maxlines and minlines
        try:
            # Generate segmented trends
            seg_trends = segtrends(recent_data, field='close', segments=max_segments + 1)
            
            # Map all the individual max and min lines to the dataframe
            last_idx = len(dataframe) - len(recent_data)
            for i in range(len(seg_trends)):
                current_idx = last_idx + i
                if current_idx < len(dataframe):
                    # Copy all available Max_Line_X and Min_Line_X columns
                    for j in range(max_segments):
                        max_col = f'Max_Line_{j}'
                        min_col = f'Min_Line_{j}'
                        
                        if max_col in seg_trends.columns:
                            dataframe.loc[current_idx, max_col] = seg_trends[max_col].iloc[i]
                        
                        if min_col in seg_trends.columns:
                            dataframe.loc[current_idx, min_col] = seg_trends[min_col].iloc[i]
        except Exception as e:
            # Fail gracefully if segtrends fails
            print(f"Error in segtrends: {e}")
        
        # Rank trendlines based on proximity to price
        try:
            # Use the rank_trendlines function to score and rank the trendlines
            trendline_rankings = rank_trendlines(
                seg_trends, 
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                touch_weight=self.trendline_touch_weight.value,
                max_prefix="Max_Line_", 
                min_prefix="Min_Line_"
            )
            
            # Store the top ranked maxlines and minlines in the dataframe
            # Add columns for the rankings
            ranked_maxlines = trendline_rankings["ranked_maxlines"]
            ranked_minlines = trendline_rankings["ranked_minlines"]
            
            # Store the ranking information in the dataframe
            # First, create columns for the top 3 ranked lines
            for rank in range(1, 4):  # Top 3 ranks
                if len(ranked_maxlines) >= rank:
                    max_col_name = list(ranked_maxlines.keys())[rank-1]
                    max_score = ranked_maxlines[max_col_name]
                    dataframe[f'Ranked_Max_{rank}'] = np.nan
                    dataframe[f'Ranked_Max_{rank}_Score'] = np.nan
                    
                    # Map the values from the ranked maxline to the new column
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(seg_trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            dataframe.loc[current_idx, f'Ranked_Max_{rank}'] = seg_trends[max_col_name].iloc[i]
                            dataframe.loc[current_idx, f'Ranked_Max_{rank}_Score'] = max_score
                
                if len(ranked_minlines) >= rank:
                    min_col_name = list(ranked_minlines.keys())[rank-1]
                    min_score = ranked_minlines[min_col_name]
                    dataframe[f'Ranked_Min_{rank}'] = np.nan
                    dataframe[f'Ranked_Min_{rank}_Score'] = np.nan
                    
                    # Map the values from the ranked minline to the new column
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(seg_trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            dataframe.loc[current_idx, f'Ranked_Min_{rank}'] = seg_trends[min_col_name].iloc[i]
                            dataframe.loc[current_idx, f'Ranked_Min_{rank}_Score'] = min_score
        
        except Exception as e:
            # Fail gracefully if ranking fails
            print(f"Error in ranking trendlines: {e}")
        
        # Mark high and low points for visualization
        try:
            # Use the high and low values instead of calculated swing points
            highest_points = recent_data.sort_values('high', ascending=False).head(20)
            lowest_points = recent_data.sort_values('low', ascending=True).head(20)
            
            for idx in highest_points.index:
                actual_idx = len(dataframe) - len(recent_data) + (idx - recent_data.index[0])
                if actual_idx < len(dataframe):
                    dataframe.loc[actual_idx, 'all_highs'] = highest_points.loc[idx, 'high']
                    
            for idx in lowest_points.index:
                actual_idx = len(dataframe) - len(recent_data) + (idx - recent_data.index[0])
                if actual_idx < len(dataframe):
                    dataframe.loc[actual_idx, 'all_lows'] = lowest_points.loc[idx, 'low']
        except Exception as e:
            print(f"Error marking high/low points: {e}")

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