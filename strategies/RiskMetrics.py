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
from scipy.stats import norm, t


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from risk_metrics.volatility_models import VolatilityModel, VolatilityRegime, GARCHModel
from risk_metrics.risk_indicators import RiskIndicators
from trend_metrics.trend_analysis import TrendAnalysis
from trend_metrics.trendline import gentrends, segtrends, rank_trendlines
from technical.util import resample_to_interval, resampled_merge

class RiskMetrics(IStrategy):
    """
    RiskMetrics strategy using HAR-RV (Heterogeneous Autoregression Realized Volatility) model
    for volatility forecasting and risk management.
    
    Features:
    - Volatility forecasting using HAR-RV model
    - Risk-adjusted position sizing based on volatility regime
    - Trendline analysis with support and resistance identification
    - Trendline ranking based on price proximity and touch frequency
    - Linear regression trendlines using TA-Lib's LINEARREG functions
      * Provides straight-line trendlines using the least squares method
      * Visualizes slope, angle, and projected forecasts
      * Useful for identifying short to medium-term trends
    - Dynamic timeframe selection based on available data
      * Automatically selects the highest appropriate timeframe
      * Adapts analysis based on available historical data length
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
    
    # Timeframe thresholds for resampling
    # Minimum number of candles needed for each timeframe
    TIMEFRAME_THRESHOLDS = {
        '1d': CANDLES_PER_DAY,           # Need at least 1 day of data
        '3d': CANDLES_PER_DAY * 3,       # Need at least 3 days of data
        '1w': WEEKLY_CANDLES,            # Need at least 1 week of data
        '1M': MONTHLY_CANDLES,           # Need at least 1 month of data
    }
    
    # Supported higher timeframes in order of preference (highest first)
    HIGHER_TIMEFRAMES = ['1M', '1w', '3d', '1d']
    
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
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

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
                "all_highs": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 12, "fillcolor": "red"},
                "all_lows": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 12, "fillcolor": "green"},
                "Max Line": {"color": "red", "width": 2.0},
                "Min Line": {"color": "green", "width": 2.0},
                "Highest_Scored_Line": {"color": "purple", "width": 3.0},
                "linear_reg_line": {"color": "blue", "width": 3.0}
            },
            "subplots": {
                "ATR": {
                    "atr": {"color": "blue"}
                },
                "Mean Scores": {
                    "Max_Mean_Score": {"color": "red", "type": "line", "width": 2.0},
                    "Min_Mean_Score": {"color": "green", "type": "line", "width": 2.0},
                    "Highest_Line_Score": {"color": "purple", "type": "line", "width": 2.5}
                },
                "Total Line Scores": {
                    "Max_Line_Score": {"color": "red", "type": "line", "width": 2.0},
                    "Min_Line_Score": {"color": "green", "type": "line", "width": 2.0}
                },
                "Timeframes": {
                    "highest_timeframe_indicator": {"color": "blue", "type": "line", "width": 2.0}
                },
                "Risk Metrics": {
                    "risk_multiplier": {"color": "orange", "type": "line", "width": 2.0}
                }
            }
        }
        
        # Dynamically add higher timeframe plots based on available timeframes
        # These won't be added until determine_highest_timeframe is called
        if hasattr(self, 'highest_timeframe') and self.highest_timeframe != self.timeframe:
            suffix = f"_{self.highest_timeframe}"
            
            # Add higher timeframe volatility to subplots
            if "Risk Metrics" not in plot_config["subplots"]:
                plot_config["subplots"]["Risk Metrics"] = {}
                
            plot_config["subplots"]["Risk Metrics"][f"rv{suffix}"] = {
                "color": "red", 
                "type": "line", 
                "width": 2.0
            }
            
            # Add higher timeframe price data to main plot with semi-transparent color
            for field in ['open', 'high', 'low', 'close']:
                field_name = f"{field}{suffix}"
                plot_config["main_plot"][field_name] = {
                    "color": "rgba(100, 100, 255, 0.3)",  # Semi-transparent blue
                    "width": 1.0
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
        # Initialize highest timeframe as None - will be determined dynamically
        self.highest_timeframe = None
        self.available_timeframes = []

    def determine_highest_timeframe(self, dataframe: DataFrame) -> str:
        """
        Determine the highest possible timeframe based on available data length.
        
        Args:
            dataframe: DataFrame with current timeframe's OHLCV data
            
        Returns:
            str: The highest timeframe that can be used ('1M', '1w', '3d', '1d' or base timeframe)
        """
        data_length = len(dataframe)
        returned_tf = []

        # Log available data
        print(f"Available data: {data_length} candles at {self.timeframe} timeframe")
                
        # Check each higher timeframe from highest to lowest
        for tf in self.HIGHER_TIMEFRAMES:
            threshold = self.TIMEFRAME_THRESHOLDS.get(tf, 0)
            print(f"Threshold for {tf}: {threshold}")
            if data_length >= threshold:
                # Add to available timeframes
                self.available_timeframes.append(tf)
                
                if returned_tf == []:
                    # Return the highest timeframe (first one that matches)
                    returned_tf = tf
        
        if returned_tf == []:
            # If no higher timeframe has enough data, return the base timeframe
            print(f"Not enough data for higher timeframes. Using base timeframe: {self.timeframe}")
            return self.timeframe
        else:
            return returned_tf

    def resample_to_higher_timeframes(self, dataframe: DataFrame) -> Dict[str, DataFrame]:
        """
        Resample the dataframe to all available higher timeframes.
        
        Args:
            dataframe: DataFrame with current timeframe's OHLCV data
            
        Returns:
            Dict[str, DataFrame]: Dictionary of resampled dataframes keyed by timeframe
        """
        resampled_dfs = {}
        
        # Skip if no data
        if len(dataframe) == 0:
            return resampled_dfs
        
        print(f"Available timeframes: {self.available_timeframes}")
        # For each available higher timeframe, resample the data
        for tf in self.available_timeframes:
            if tf == self.timeframe:
                continue  # Skip the base timeframe
                
            try:
                # Use resample_to_interval to convert to higher timeframe
                resampled = resample_to_interval(dataframe, self.TIMEFRAME_THRESHOLDS[tf])
                
                # Add to our dictionary
                resampled_dfs[tf] = resampled
                
                print(f"Successfully resampled to {tf} timeframe: {len(resampled)} candles")
            except Exception as e:
                print(f"Error resampling to {tf}: {e}")
        
        return resampled_dfs

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
        # We'll dynamically determine the informative pairs based on the highest timeframe
        # This is done at runtime in populate_indicators
        return []

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame, including ATR.
        """
        if len(dataframe) == 0:
            return dataframe
            
        # Determine the highest timeframe we can use based on available data
        self.highest_timeframe = self.determine_highest_timeframe(dataframe)
        
        # Add an indicator showing which highest timeframe was selected
        dataframe['highest_timeframe_indicator'] = 1.0
        dataframe['highest_timeframe'] = self.highest_timeframe
        print(f"Highest timeframe: {self.highest_timeframe}")
        # Resample to higher timeframes if possible
        resampled_dfs = self.resample_to_higher_timeframes(dataframe)
        print(f"Resampled dataframes: {resampled_dfs}")

        # Calculate ATR and store it for visualization
        dataframe['atr'] = self.volatility_model.calculate_atr(dataframe)

        # Initialize marker columns for support and resistance points
        dataframe['all_highs'] = np.nan
        dataframe['all_lows'] = np.nan
        
        # Initialize columns for main trend lines
        dataframe['Max Line'] = np.nan
        dataframe['Min Line'] = np.nan
        
        # Initialize new columns for highest scored line
        dataframe['Highest_Scored_Line'] = np.nan
        dataframe['Highest_Set_Mean'] = np.nan
        dataframe['Highest_Line_Score'] = np.nan
        dataframe['Highest_Line_Type'] = ""  # Will be "Resistance" or "Support"
        dataframe['Highest_Line_Text'] = ""  # For displaying text on the chart
        
        # Define the maximum number of segments we'll use
        max_segments = 10
        
        # Initialize columns for all individual segment trend lines
        for i in range(max_segments):
            dataframe[f'Max_Line_{i}'] = np.nan
            dataframe[f'Min_Line_{i}'] = np.nan
        
        # Calculate trendlines using local maxima/minima
        lookback = 200  # Use last 200 candles for trendline calculation

        # Don't calculate trendlines if we don't have enough data
        if len(dataframe) < 30:
            return dataframe
            
        recent_data = dataframe.tail(lookback).copy()
        
        # Calculate linear regression trendlines using TA-Lib
        # These will provide straight line trendlines using the least squares method
        try:
            # Get appropriate timeperiod for linear regression
            timeperiod = self.linearreg_timeperiod.value
            price_field = self.linearreg_price_field.value
            
            # The linear regression trendline is a true straight line calculated using
            # the least squares fit method over the specified timeperiod. Unlike the other
            # trendlines that connect pivot points, this line represents the statistical
            # best fit line through the price data and can help identify the trend
            # direction and strength. A steeper slope indicates a stronger trend.
            linearreg_data = self.trend_analyzer.calculate_talib_linearreg(
                dataframe, 
                timeperiod=timeperiod,
                price_field=price_field
            )
            
            # Copy the linear regression columns back to the original dataframe
            dataframe['linear_reg'] = linearreg_data['linear_reg']
            dataframe['linear_reg_slope'] = linearreg_data['linear_reg_slope']
            dataframe['linear_reg_angle'] = linearreg_data['linear_reg_angle']
            dataframe['linear_reg_intercept'] = linearreg_data['linear_reg_intercept']
            dataframe['linear_reg_line'] = linearreg_data['linear_reg_line']
                    
        except Exception as e:
            print(f"Error calculating linear regression: {e}")

        # Mark high and low points for visualization
        try:
            # Use the TrendAnalysis._find_swing_points function to find significant swing points
            # This provides better identification of true support and resistance levels
            high_swing_points = self.trend_analyzer._find_swing_points(
                prices=recent_data['high'].values,
                price_type='high',
                min_points=5,
                distance=10
            )
            
            low_swing_points = self.trend_analyzer._find_swing_points(
                prices=recent_data['low'].values,
                price_type='low',
                min_points=5,
                distance=10
            )
            
            # Initialize all_highs and all_lows in recent_data with NaN values
            recent_data['all_highs'] = np.nan
            recent_data['all_lows'] = np.nan
            
            # Map swing high points to the dataframe
            for idx, price in high_swing_points:
                if idx < len(recent_data):
                    actual_idx = len(dataframe) - len(recent_data) + idx
                    if 0 <= actual_idx < len(dataframe):
                        dataframe.loc[actual_idx, 'all_highs'] = price
                        recent_data.iloc[idx, recent_data.columns.get_loc('all_highs')] = price
            
            # Map swing low points to the dataframe
            for idx, price in low_swing_points:
                if idx < len(recent_data):
                    actual_idx = len(dataframe) - len(recent_data) + idx
                    if 0 <= actual_idx < len(dataframe):
                        dataframe.loc[actual_idx, 'all_lows'] = price
                        recent_data.iloc[idx, recent_data.columns.get_loc('all_lows')] = price
            
        except Exception as e:
            print(f"Error finding swing points: {e}")

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
                min_prefix="Min_Line_",
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=8.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Store the top ranked maxlines and minlines in the dataframe
            # Add columns for the rankings
            ranked_maxlines = trendline_rankings["ranked_maxlines"]
            ranked_minlines = trendline_rankings["ranked_minlines"]
            
            # Calculate mean scores for each set and select the highest scored line
            scores_result = self.trend_analyzer.calculate_trendline_set_scores(ranked_maxlines, ranked_minlines)
            
            # Store mean scores in the dataframe
            max_mean_score = scores_result["max_mean_score"]
            min_mean_score = scores_result["min_mean_score"]
            highest_set = scores_result["highest_set"]
            highest_line_name = scores_result["highest_line_name"]
            highest_line_score = scores_result["highest_line_score"]
            
            # Add indicators for visualization
            dataframe['Max_Mean_Score'] = max_mean_score
            dataframe['Min_Mean_Score'] = min_mean_score
            
            # Calculate and store scores for the main Max Line and Min Line
            main_lines_score = rank_trendlines(
                trends,  # Use the gentrends output that has Max Line and Min Line
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                touch_weight=self.trendline_touch_weight.value,
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=9.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Extract the scores for Max Line and Min Line
            max_line_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
            min_line_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
            
            # Store the scores in the dataframe
            dataframe['Max_Line_Score'] = max_line_score
            dataframe['Min_Line_Score'] = min_line_score
                        
            # Copy the highest scored line to the Highest_Scored_Line column
            if highest_line_name is not None:
                last_idx = len(dataframe) - len(recent_data)
                for i in range(len(seg_trends)):
                    current_idx = last_idx + i
                    if current_idx < len(dataframe):
                        dataframe.loc[current_idx, 'Highest_Scored_Line'] = seg_trends[highest_line_name].iloc[i]
                        dataframe.loc[current_idx, 'Highest_Set_Mean'] = max_mean_score if highest_set == "max" else min_mean_score
                        dataframe.loc[current_idx, 'Highest_Line_Score'] = highest_line_score
                        dataframe.loc[current_idx, 'Highest_Line_Type'] = "Resistance" if highest_set == "max" else "Support"
                        
        except Exception as e:
            # Fail gracefully if ranking fails
            print(f"Error in ranking trendlines: {e}")

        print("--------------------------------")
        print("--------------------------------")
        print("Begin Default GARCH")        
        print("---")

        # Exemple d'utilisation
        garch_model = GARCHModel()
        risk_indicators = RiskIndicators()

        # Example 1: Basic GARCH(1,1) with default parameters

        simulated_returns = [0.07, 0.06, 0.05, 0.09]
        np_array = np.array(simulated_returns)
        # Calculate log returns on arithmetics returns
        log_returns = np.log(1 + np_array)
        # Calculate log returns based on price levels
        #log_returns = np.log(prices[1:] / prices[:-1])

        #df = 5  # Can be adjusted based on empirical data
        #VaR_Z = t.ppf(1 - confidence_level, df) * std
        confidence_level = 0.01
        VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
            simulated_returns=log_returns,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(log_returns)
        )
        garch_model.reset_variance()
        print(f"VaR à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
        print(f"ES à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")

        print("--------------------------------")
        print("Begin Monte Carlo with GARCH")        
        print("---")

        # Example 2: Basic GARCH(1,1) with Monte Carlo
        simulated_returns = garch_model.monte_carlo_simulation(T=3, iterations=1000)
        confidence_level = 0.01
        VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
            simulated_returns=simulated_returns,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(simulated_returns)
        )
        print(f"VaR à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
        print(f"ES à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")
            
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

    def get_market_condition_description(self, dataframe: DataFrame) -> Dict[str, str]:
        """
        Generate a human-readable description of current market conditions
        based on the timeframe analysis.
        
        Args:
            dataframe: The analyzed dataframe with indicators
            
        Returns:
            A dictionary containing market condition descriptions
        """
        if len(dataframe) < 10:
            return {"error": "Not enough data for market condition analysis"}
            
        # Get the most recent candle
        current_candle = dataframe.iloc[-1].squeeze()
        
        # Get the highest timeframe used
        highest_tf = current_candle.get('highest_timeframe', self.timeframe)
        
        # Get volatility regime
        volatility_regime = current_candle.get('volatility_regime', 'Unknown')
        
        # Get risk multiplier
        risk_multiplier = current_candle.get('risk_multiplier', 1.0)
        
        # Get linear regression angle to determine trend direction
        trend_angle = current_candle.get('linear_reg_angle', 0)
        
        # Determine trend direction based on angle
        if trend_angle > 45:
            trend_direction = "Strong uptrend"
        elif trend_angle > 20:
            trend_direction = "Moderate uptrend"
        elif trend_angle > 5:
            trend_direction = "Mild uptrend"
        elif trend_angle > -5:
            trend_direction = "Sideways"
        elif trend_angle > -20:
            trend_direction = "Mild downtrend"
        elif trend_angle > -45:
            trend_direction = "Moderate downtrend"
        else:
            trend_direction = "Strong downtrend"
        
        # Get highest scored line type (support/resistance)
        strongest_level_type = current_candle.get('Highest_Line_Type', 'Unknown')
        
        # Format the percentage of risk based on the multiplier
        risk_percentage = risk_multiplier * 100
        
        # Create description
        description = {
            "timeframe": f"Analysis based on {highest_tf} data",
            "volatility": f"{volatility_regime} volatility environment",
            "trend": trend_direction,
            "important_level": f"Most significant level: {strongest_level_type}",
            "risk_assessment": f"Recommended position size: {risk_percentage:.1f}% of maximum",
            "summary": (
                f"Market is in a {trend_direction.lower()} with {volatility_regime.lower()} volatility. "
                f"Position sizing set to {risk_percentage:.1f}% based on {highest_tf} analysis."
            )
        }
        
        return description