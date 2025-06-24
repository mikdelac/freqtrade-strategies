# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from functools import reduce
from scipy.stats import norm, t
import random


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
from risk_metrics.monte_carlo import MonteCarloSimulator, MonteCarloManager
from trend_metrics.trend_analysis import TrendAnalysis
from trend_metrics.trendline import gentrends, segtrends, rank_trendlines, generate_bounce_conditions
from technical.util import resample_to_interval, resampled_merge

class SignalGenerator:
    """
    Signal generation logic for bounce trading with convergence detection.
    Handles entry and exit signal generation with proper separation of concerns.
    """
    
    def __init__(self, strategy_instance):
        """
        Initialize SignalGenerator with reference to strategy instance.
        
        Args:
            strategy_instance: Reference to the main RiskMetrics strategy
        """
        self.strategy = strategy_instance
        
    def apply_convergence_filter(self, dataframe: DataFrame, conditions: pd.Series, 
                               enable_convergence: bool, threshold: float) -> pd.Series:
        """
        Apply convergence filter to trading conditions.
        
        Args:
            dataframe: DataFrame with MC scores
            conditions: Boolean series of trading conditions
            enable_convergence: Whether convergence detection is enabled
            threshold: Convergence threshold for filtering
            
        Returns:
            pandas.Series: Filtered conditions with convergence applied
        """
        if not enable_convergence:
            return conditions
            
        # Calculate convergence ratios for each row
        convergence_ratios = dataframe.apply(
            lambda row: self.strategy.calculate_score_convergence_ratio(
                row.get('MC_Support_Score', 0), 
                row.get('MC_Resistance_Score', 0)
            ), axis=1
        )
        
        # Apply convergence filter
        convergence_filter = convergence_ratios < threshold
        return conditions & convergence_filter
    
    def generate_entry_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Generate entry signals for both long and short positions.
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame: Updated dataframe with entry signals
        """
        # Initialize entry signals
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Check if required columns exist
        required_columns = ['MC_Optimal_Support', 'MC_Optimal_Resistance', 
                          'MC_Support_Score', 'MC_Resistance_Score']
        if not all(col in dataframe.columns for col in required_columns):
            print("Missing required columns for signal generation")
            return dataframe
        
        # Additional validation: Check if trendlines have valid values
        # Create a mask for rows where both support and resistance have valid values
        valid_support = ~dataframe['MC_Optimal_Support'].isna()
        valid_resistance = ~dataframe['MC_Optimal_Resistance'].isna()
        valid_trendlines = valid_support & valid_resistance
        
        if not valid_trendlines.any():
            print("No valid trendline data available - skipping entry signal generation")
            return dataframe
        
        # Generate long entry conditions (bounce off support) - using direct import from trendline.py
        long_bounce_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Support'], 'long', self.strategy.trendline_proximity_threshold.value,
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        
        # Apply convergence filter for long entries
        long_conditions_filtered = self.apply_convergence_filter(
            dataframe, long_bounce_conditions, 
            self.strategy.enable_convergence_detection.value,
            self.strategy.score_convergence_high_threshold.value
        )
        
        # Apply trendline validity filter for long entries
        long_conditions_filtered = long_conditions_filtered & valid_trendlines
        
        # Generate short entry conditions (bounce off resistance) - using direct import from trendline.py
        short_bounce_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy.trendline_proximity_threshold.value,
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        
        # Apply convergence filter for short entries
        short_conditions_filtered = self.apply_convergence_filter(
            dataframe, short_bounce_conditions,
            self.strategy.enable_convergence_detection.value,
            self.strategy.score_convergence_high_threshold.value
        )
        
        # Apply trendline validity filter for short entries
        short_conditions_filtered = short_conditions_filtered & valid_trendlines
        
        # Set entry signals
        dataframe.loc[long_conditions_filtered, 'enter_long'] = 1
        dataframe.loc[short_conditions_filtered, 'enter_short'] = 1
        
        # Log entry signal summary
        long_signals = dataframe['enter_long'].sum()
        short_signals = dataframe['enter_short'].sum()
        valid_rows = valid_trendlines.sum()
        total_rows = len(dataframe)
        
        print(f"Entry signals generated: {long_signals} long, {short_signals} short")
        print(f"Valid trendline data: {valid_rows}/{total_rows} rows ({valid_rows/total_rows*100:.1f}%)")
        
        # Log convergence analysis for the most recent candle
        if self.strategy.enable_convergence_detection.value and len(dataframe) > 0:
            current_candle = dataframe.iloc[-1]
            support_score = current_candle.get('MC_Support_Score', 0)
            resistance_score = current_candle.get('MC_Resistance_Score', 0)
            
            should_enter, reason = self.strategy.should_enter_trade_with_convergence(
                support_score, resistance_score
            )
            print(f"Current convergence status: {reason}")
            print(f"Entry allowed: {should_enter}")
            
            # Check if current candle has valid trendlines
            current_valid = valid_trendlines.iloc[-1] if len(valid_trendlines) > 0 else False
            print(f"Current candle trendline validity: {current_valid}")
        
        return dataframe
    
    def generate_break_exit_conditions(self, dataframe: DataFrame, level_column: str, 
                                     direction: str) -> pd.Series:
        """
        Generate exit conditions based on support/resistance breaks.
        
        Args:
            dataframe: DataFrame with OHLCV data
            level_column: Column name for support/resistance level
            direction: Either 'long' for support break or 'short' for resistance break
            
        Returns:
            pandas.Series: Boolean series indicating break exit conditions
        """
        if level_column not in dataframe.columns:
            return pd.Series([False] * len(dataframe), index=dataframe.index)
        
        conviction_threshold = 0.002  # 0.2% conviction threshold
        
        if direction == 'long':
            # Exit long when support is definitively broken
            break_conditions = (
                (dataframe['close'] < dataframe[level_column]) &
                (dataframe['close'].shift(1) >= dataframe[level_column].shift(1)) &
                # Add conviction: close is significantly below support
                (dataframe['close'] < dataframe[level_column] * (1 - conviction_threshold)) &
                (~dataframe[level_column].isna()) &
                (~dataframe[level_column].shift(1).isna())
            )
        elif direction == 'short':
            # Exit short when resistance is definitively broken
            break_conditions = (
                (dataframe['close'] > dataframe[level_column]) &
                (dataframe['close'].shift(1) <= dataframe[level_column].shift(1)) &
                # Add conviction: close is significantly above resistance
                (dataframe['close'] > dataframe[level_column] * (1 + conviction_threshold)) &
                (~dataframe[level_column].isna()) &
                (~dataframe[level_column].shift(1).isna())
            )
        else:
            return pd.Series([False] * len(dataframe), index=dataframe.index)
            
        return break_conditions
    
    def generate_cross_signal_exits(self, dataframe: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate cross-signal exits (exit long on short conditions, exit short on long conditions).
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            Tuple[pd.Series, pd.Series]: (exit_long_conditions, exit_short_conditions)
        """
        # Generate short entry conditions for long exits - using direct import from trendline.py
        short_entry_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy.trendline_proximity_threshold.value,
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        
        # Generate long entry conditions for short exits - using direct import from trendline.py
        long_entry_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Support'], 'long', self.strategy.trendline_proximity_threshold.value,
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        
        # Apply convergence filter if enabled
        if self.strategy.enable_convergence_detection.value:
            short_entry_filtered = self.apply_convergence_filter(
                dataframe, short_entry_conditions,
                True, self.strategy.score_convergence_high_threshold.value
            )
            long_entry_filtered = self.apply_convergence_filter(
                dataframe, long_entry_conditions,
                True, self.strategy.score_convergence_high_threshold.value
            )
        else:
            short_entry_filtered = short_entry_conditions
            long_entry_filtered = long_entry_conditions
        
        return short_entry_filtered, long_entry_filtered
    
    def generate_exit_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Generate exit signals for both long and short positions.
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame: Updated dataframe with exit signals
        """
        # Initialize exit signals
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # === Traditional Support/Resistance Break Exits ===
        
        # Exit long when support is broken
        if 'MC_Optimal_Support' in dataframe.columns:
            support_break_exit = self.generate_break_exit_conditions(
                dataframe, 'MC_Optimal_Support', 'long'
            )
            dataframe.loc[support_break_exit, 'exit_long'] = 1
        
        # Exit short when resistance is broken
        if 'MC_Optimal_Resistance' in dataframe.columns:
            resistance_break_exit = self.generate_break_exit_conditions(
                dataframe, 'MC_Optimal_Resistance', 'short'
            )
            dataframe.loc[resistance_break_exit, 'exit_short'] = 1
        
        # === Cross-Signal Exits ===
        
        # Check if required columns exist for cross-signal exits
        required_columns = ['MC_Optimal_Support', 'MC_Optimal_Resistance', 
                          'MC_Support_Score', 'MC_Resistance_Score']
        if all(col in dataframe.columns for col in required_columns):
            exit_long_cross, exit_short_cross = self.generate_cross_signal_exits(dataframe)
            
            # Apply cross-signal exits
            dataframe.loc[exit_long_cross, 'exit_long'] = 1
            dataframe.loc[exit_short_cross, 'exit_short'] = 1
        
        # Log exit signal summary
        long_exits = dataframe['exit_long'].sum()
        short_exits = dataframe['exit_short'].sum()
        print(f"Exit signals generated: {long_exits} long exits, {short_exits} short exits")
        
        return dataframe

class RiskMetrics(IStrategy):
    """
    RiskMetrics strategy using proper GARCH implementation for volatility forecasting and risk management.
    
    Features:
    - GARCH model for volatility forecasting and risk management (properly separated from technical analysis)
    - Risk-adjusted position sizing based on volatility regime
    - Fixed lookback periods for Monte Carlo optimization (not volatility-based)
    - Trendline analysis with support and resistance identification
    - Trendline ranking based on price proximity and touch frequency
    - Monte Carlo optimization for optimal trendline periods using fixed sampling
    - Linear regression trendlines using TA-Lib's LINEARREG functions
      * Provides straight-line trendlines using the least squares method
      * Visualizes slope, angle, and projected forecasts
      * Useful for identifying short to medium-term trends
    - Dynamic timeframe selection based on available data
      * Automatically selects the highest appropriate timeframe
      * Adapts analysis based on available historical data length
    - Monte Carlo Score Convergence Detection and Risk Management
      * Detects when support and resistance scores are similar
      * Implements adaptive position sizing during convergence periods
      * Switches between bounce trading and breakout modes
    
    GARCH Usage:
    - Estimates current market volatility for risk management
    - Provides volatility regime classification (low/medium/high)
    - Used for position sizing and risk multipliers
    """
    INTERFACE_VERSION = 3

    # Timeframe settings
    timeframe = "1m"
    MINUTES_IN_DAY = 24 * 60
    MINUTES_PER_CANDLE = 1
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
    
    # Monte Carlo period optimization settings
    MC_ITERATIONS = 200
    MIN_LOOKBACK_PERIOD = 50
    # MAX_LOOKBACK_PERIOD will be set dynamically based on available data
    
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
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.0005, space="buy", optimize=True)
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

    # Monte Carlo optimization parameters
    enable_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)

    # GARCH Volatility-based period sampling parameters
    # These parameters were incorrectly mixing GARCH volatility estimation with lookback period selection
    # GARCH is now properly used only for risk management and volatility forecasting
    garch_min_candles = IntParameter(100, 300, default=100, space="buy", optimize=False)  # Kept for compatibility
    garch_max_candles = IntParameter(500, 1000, default=1000, space="buy", optimize=False)  # Kept for compatibility
    
    
    # Monte Carlo Score Convergence Parameters
    score_convergence_high_threshold = DecimalParameter(0.85, 0.95, default=0.90, space="buy", optimize=True)
    score_convergence_medium_threshold = DecimalParameter(0.70, 0.85, default=0.80, space="buy", optimize=True)
    score_convergence_low_threshold = DecimalParameter(0.50, 0.70, default=0.60, space="buy", optimize=True)
    
    # Convergence risk multipliers
    convergence_high_penalty = DecimalParameter(0.1, 0.3, default=0.2, space="buy", optimize=True)
    convergence_medium_penalty = DecimalParameter(0.4, 0.6, default=0.5, space="buy", optimize=True)
    convergence_low_penalty = DecimalParameter(0.7, 0.9, default=0.8, space="buy", optimize=True)
    
    # Enable convergence detection
    enable_convergence_detection = BooleanParameter(default=False, space="buy", optimize=False)
        
    # === New Periodic Monte Carlo Parameters ===
    mc_recalc_interval_minutes = IntParameter(60, 480, default=MINUTES_PER_CANDLE * 30, space="buy", optimize=False)
    mc_lookback_window_candles = IntParameter(1000, 3000, default=2000, space="buy", optimize=False)
    
    # Rolling Monte Carlo optimization (eliminates lookahead bias)
    enable_rolling_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "360": 0.15,  # Exit after 6 hours if profit is 15%
        "240": 0.10,  # Exit after 4 hours if profit is 10%
        "120": 0.07,  # Exit after 2 hours if profit is 7%
        "60": 0.05,   # Exit after 1 hour if profit is 5%
        "30": 0.03,   # Exit after 30 min if profit is 3%
        "0": 0.02     # Exit immediately if profit is 2%
    }

    # Stoploss configuration - completely disable trailing stops
    stoploss = -0.1  # Fallback value if custom_stoploss fails
    trailing_stop = False
    trailing_stop_positive = None  # Explicitly disable
    trailing_stop_positive_offset = None  # Explicitly disable
    trailing_only_offset_is_reached = None  # Explicitly disable
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
        "stoploss_on_exchange": True
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    def calculate_bounce_metrics(self, dataframe: DataFrame) -> Dict[str, float]:
        """
        Calculate bounce counts and related metrics for MC_Optimal_Resistance and MC_Optimal_Support.
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            Dict containing bounce metrics for both resistance and support
        """
        metrics = {
            'resistance_bounce_count': 0.0,
            'support_bounce_count': 0.0,
            'resistance_score': 0.0,
            'support_score': 0.0
        }
        
        # Check if required columns exist
        required_columns = ['MC_Optimal_Support', 'MC_Optimal_Resistance', 
                          'MC_Support_Score', 'MC_Resistance_Score']
        if not all(col in dataframe.columns for col in required_columns):
            return metrics
        
        # Check if pivot points are available
        if 'all_highs' not in dataframe.columns or 'all_lows' not in dataframe.columns:
            return metrics
        
        try:
            # Calculate resistance bounces (short direction)
            resistance_bounce_conditions = generate_bounce_conditions(
                dataframe['close'], 
                dataframe['MC_Optimal_Resistance'], 'short', self.trendline_proximity_threshold.value,
                dataframe.get('all_highs'), dataframe.get('all_lows')
            )
            metrics['resistance_bounce_count'] = float(resistance_bounce_conditions.sum())
            
            # Calculate support bounces (long direction)  
            support_bounce_conditions = generate_bounce_conditions(
                dataframe['close'], 
                dataframe['MC_Optimal_Support'], 'long', self.trendline_proximity_threshold.value,
                dataframe.get('all_highs'), dataframe.get('all_lows')
            )
            metrics['support_bounce_count'] = float(support_bounce_conditions.sum())
            
            # Get the latest scores
            if len(dataframe) > 0:
                latest_candle = dataframe.iloc[-1]
                metrics['resistance_score'] = float(latest_candle.get('MC_Resistance_Score', 0.0))
                metrics['support_score'] = float(latest_candle.get('MC_Support_Score', 0.0))
                
        except Exception as e:
            print(f"Error calculating bounce metrics: {e}")
        
        return metrics

    @property
    def plot_config(self):
        # Basic configuration with default plots
        plot_config = {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 12, "fillcolor": "red"},
                "all_lows": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 12, "fillcolor": "green"},
                "Resistance Line": {"color": "red", "width": 2.0},
                "Support Line": {"color": "green", "width": 2.0},
                "Highest_Scored_Line": {"color": "purple", "width": 3.0},
                "linear_reg_line": {"color": "blue", "width": 3.0},
                "MC_Optimal_Resistance": {"color": "darkred", "width": 4.0, "dash": "dot"},
                "MC_Optimal_Support": {"color": "darkgreen", "width": 4.0, "dash": "dot"}
            },
            "subplots": {
                "Total Line Scores": {
                    "Resistance_Line_Score": {"color": "red", "type": "line", "width": 2.0},
                    "Support_Line_Score": {"color": "green", "type": "line", "width": 2.0}
                },
                "Monte Carlo Optimization": {
                    "MC_Resistance_Score": {"color": "darkred", "type": "line", "width": 3.0},
                    "MC_Support_Score": {"color": "darkgreen", "type": "line", "width": 3.0},
                    "MC_Optimal_Period": {"color": "orange", "type": "line", "width": 1.5}
                },
                "Bounce Analysis": {
                    "resistance_bounce_count": {"color": "darkred", "type": "line", "width": 2.0},
                    "support_bounce_count": {"color": "darkgreen", "type": "line", "width": 2.0},
                    "resistance_bounce_score_display": {"color": "red", "type": "line", "width": 1.5, "dash": "dash"},
                    "support_bounce_score_display": {"color": "green", "type": "line", "width": 1.5, "dash": "dash"}
                },
                "Score Convergence Analysis": {
                    "score_convergence_ratio": {"color": "purple", "type": "line", "width": 2.0},
                    "convergence_multiplier": {"color": "orange", "type": "line", "width": 2.0}
                },
                "Trading Mode": {
                    "trading_mode_indicator": {"color": "blue", "type": "line", "width": 2.0}
                }
            }
        }
        
        # Add volatility regime subplot (always enabled for proper GARCH risk management)
        plot_config["subplots"]["Volatility Analysis"] = {
            "volatility": {"color": "purple", "type": "line", "width": 2.0},
            "risk_multiplier": {"color": "orange", "type": "line", "width": 2.0},
            "atr": {"color": "cyan", "type": "line", "width": 2.0}
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
        
        # Initialize GARCH model for risk management
        self.garch_model = GARCHModel()
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(self)
        
        # Initialize Monte Carlo Manager with all required parameters
        self.monte_carlo_manager = MonteCarloManager(
            mc_iterations=self.MC_ITERATIONS,
            min_lookback_period=self.MIN_LOOKBACK_PERIOD,
            recalc_interval_minutes=self.mc_recalc_interval_minutes.value,
            trendline_proximity_threshold=self.trendline_proximity_threshold.value,
            trend_analyzer=self.trend_analyzer,
            volatility_model=self.volatility_model,
            garch_model=self.garch_model
        )
        
        print(f"RiskMetrics strategy initialized with modular Monte Carlo architecture:")
        print(f"  GARCH model: Used for risk management and volatility estimation only")
        print(f"  Lookback periods: Fixed periods for Monte Carlo optimization")
        print(f"  Monte Carlo iterations: {self.MC_ITERATIONS}")
        print(f"  Score convergence detection enabled: {self.enable_convergence_detection.value}")
        print(f"  Convergence thresholds: High={self.score_convergence_high_threshold.value:.2f}, Medium={self.score_convergence_medium_threshold.value:.2f}, Low={self.score_convergence_low_threshold.value:.2f}")
        print(f"  Proper GARCH usage: Volatility forecasting separate from technical analysis optimization")
        print(f"  Signal generator: Initialized for modular signal generation")
        print(f"  Monte Carlo Manager: Initialized with {self.mc_recalc_interval_minutes.value}min recalc interval")
        print(f"  Modular Monte Carlo: Clean separation of concerns with specialized classes")
        print(f"  Rolling Monte Carlo optimization: {'ENABLED' if self.enable_rolling_mc_optimization.value else 'DISABLED'}")
        if self.enable_rolling_mc_optimization.value:
            print(f"    - Eliminates lookahead bias for realistic backtesting")
            print(f"    - Uses {self.mc_lookback_window_candles.value} candle lookback window")
            print(f"    - Recalculates every {self.mc_recalc_interval_minutes.value} minutes")
        else:
            print(f"    - Using original method (faster but with potential lookahead bias)")

    def calculate_score_convergence_ratio(self, support_score: float, resistance_score: float) -> float:
        """
        Calculate the convergence ratio between MC support and resistance scores.
        Returns the ratio of the smaller score to the larger score (0.0 to 1.0).
        
        Args:
            support_score: MC_Support_Score value
            resistance_score: MC_Resistance_Score value
            
        Returns:
            float: Convergence ratio (0.0 = completely different, 1.0 = identical)
        """
        if support_score <= 0 or resistance_score <= 0:
            return 0.0
        
        # Calculate the ratio of smaller to larger score
        min_score = min(support_score, resistance_score)
        max_score = max(support_score, resistance_score)
        
        return min_score / max_score

    def get_convergence_multiplier(self, convergence_ratio: float) -> float:
        """
        Calculate position size multiplier based on score convergence ratio.
        Higher convergence = lower position size due to increased uncertainty.
        
        Args:
            convergence_ratio: Score convergence ratio (0.0 to 1.0)
            
        Returns:
            float: Position size multiplier (0.0 to 1.0)
        """
        if not self.enable_convergence_detection.value:
            return 1.0
        
        if convergence_ratio >= self.score_convergence_high_threshold.value:
            # High convergence - significant risk reduction
            return self.convergence_high_penalty.value
        elif convergence_ratio >= self.score_convergence_medium_threshold.value:
            # Medium convergence - moderate risk reduction
            return self.convergence_medium_penalty.value
        elif convergence_ratio >= self.score_convergence_low_threshold.value:
            # Low convergence - slight risk reduction
            return self.convergence_low_penalty.value
        
        # No significant convergence - no penalty
        return 1.0

    def get_trading_mode(self, convergence_ratio: float) -> str:
        """
        Determine current trading mode based on score convergence ratio.
        
        Args:
            convergence_ratio: Score convergence ratio (0.0 to 1.0)
            
        Returns:
            str: Trading mode identifier
        """
        if not self.enable_convergence_detection.value:
            return "NORMAL_BOUNCE"
        
        if convergence_ratio >= self.score_convergence_high_threshold.value:
            return "HIGH_CONVERGENCE"    # Avoid trading or prepare for breakout
        elif convergence_ratio >= self.score_convergence_medium_threshold.value:
            return "MEDIUM_CONVERGENCE"  # Reduced size + breakout watch
        elif convergence_ratio >= self.score_convergence_low_threshold.value:
            return "LOW_CONVERGENCE"     # Slight caution
        
        return "NORMAL_BOUNCE"  # Standard bounce trading

    def should_enter_trade_with_convergence(self, support_score: float, resistance_score: float) -> Tuple[bool, str]:
        """
        Enhanced entry logic considering score convergence.
        
        Args:
            support_score: MC_Support_Score value
            resistance_score: MC_Resistance_Score value
            
        Returns:
            Tuple[bool, str]: (should_enter, reason)
        """
        if not self.enable_convergence_detection.value:
            return True, "CONVERGENCE_DISABLED"
        
        convergence_ratio = self.calculate_score_convergence_ratio(support_score, resistance_score)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        if trading_mode == "HIGH_CONVERGENCE":
            return False, f"HIGH_CONVERGENCE_DETECTED ({convergence_ratio:.3f})"
        elif trading_mode == "MEDIUM_CONVERGENCE":
            return True, f"MEDIUM_CONVERGENCE_CAUTION ({convergence_ratio:.3f})"
        elif trading_mode == "LOW_CONVERGENCE":
            return True, f"LOW_CONVERGENCE_SLIGHT_CAUTION ({convergence_ratio:.3f})"
        
        return True, f"NORMAL_BOUNCE_MODE ({convergence_ratio:.3f})"

    def detect_breakout_scenario(self, dataframe: DataFrame) -> Dict[str, any]:
        """
        Detect when market is in breakout mode due to score convergence.
        
        Args:
            dataframe: DataFrame with MC scores
            
        Returns:
            Dict containing breakout analysis
        """
        if len(dataframe) == 0:
            return {"mode": "INSUFFICIENT_DATA", "ratio": 0.0, "multiplier": 1.0}
        
        current_candle = dataframe.iloc[-1]
        support_score = current_candle.get('MC_Support_Score', 0)
        resistance_score = current_candle.get('MC_Resistance_Score', 0)
        
        convergence_ratio = self.calculate_score_convergence_ratio(support_score, resistance_score)
        convergence_multiplier = self.get_convergence_multiplier(convergence_ratio)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        return {
            "mode": trading_mode,
            "ratio": convergence_ratio,
            "multiplier": convergence_multiplier,
            "support_score": support_score,
            "resistance_score": resistance_score,
            "recommendation": self._get_trading_recommendation(trading_mode, convergence_ratio)
        }

    def _get_trading_recommendation(self, trading_mode: str, convergence_ratio: float) -> str:
        """
        Get trading recommendation based on convergence analysis.
        
        Args:
            trading_mode: Current trading mode
            convergence_ratio: Score convergence ratio
            
        Returns:
            str: Trading recommendation
        """
        recommendations = {
            "HIGH_CONVERGENCE": f"AVOID bounce trades. Scores too similar ({convergence_ratio:.3f}). Wait for breakout or clear divergence.",
            "MEDIUM_CONVERGENCE": f"CAUTION: Reduced position size. Monitor for breakout signals. Convergence ratio: {convergence_ratio:.3f}",
            "LOW_CONVERGENCE": f"SLIGHT CAUTION: Minor convergence detected ({convergence_ratio:.3f}). Normal trading with reduced risk.",
            "NORMAL_BOUNCE": f"NORMAL bounce trading conditions. Clear score differentiation ({convergence_ratio:.3f})."
        }
        
        return recommendations.get(trading_mode, f"Unknown mode: {trading_mode}")

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
        Now implements rolling Monte Carlo optimization to eliminate lookahead bias.
        """
        if len(dataframe) == 0:
            return dataframe
            
        # Determine the highest timeframe we can use based on available data
        self.highest_timeframe = self.determine_highest_timeframe(dataframe)
        
        # Add an indicator showing which highest timeframe was selected
        dataframe.loc[:, 'highest_timeframe_indicator'] = 1.0
        dataframe.loc[:, 'highest_timeframe'] = self.highest_timeframe
        print(f"Highest timeframe: {self.highest_timeframe}")
        # Resample to higher timeframes if possible
        resampled_dfs = self.resample_to_higher_timeframes(dataframe)
        print(f"Resampled dataframes: {resampled_dfs}")

        # Calculate ATR and store it for visualization
        dataframe.loc[:, 'atr'] = self.volatility_model.calculate_atr(dataframe)

        # Calculate and store volatility regime information using separated GARCH estimation
        if len(dataframe) >= 100:  # Use fixed minimum threshold for volatility estimation
            print("=== GARCH Volatility Estimation for Risk Management ===")
            
            # Calculate log returns for GARCH volatility estimation
            log_returns = np.log(dataframe['close'].pct_change() + 1).dropna().values
            
            # Use a reasonable window for volatility estimation
            volatility_window = min(len(log_returns), 500)  # Fixed window for volatility estimation
            volatility_window = max(volatility_window, 100)   # Minimum window for reliable estimation
            
            # Calculate current volatility using GARCH model
            recent_returns = log_returns[-volatility_window:]
            
            # Clean data and calculate volatility
            if len(recent_returns) > 0 and not np.isnan(recent_returns).all() and not np.isinf(recent_returns).any():
                # Use VolatilityModel's calculate_volatility method with GARCHModel
                current_volatility = self.volatility_model.calculate_volatility(recent_returns, self.garch_model)
                print(f"  GARCH volatility result: {current_volatility:.6f}")
            else:
                print(f"  Invalid data for GARCH calculation, using fallback")
                # Fallback to simple standard deviation using VolatilityModel
                current_volatility = self.volatility_model.calculate_volatility(recent_returns)
                
            regime, risk_multiplier = self.volatility_model.get_regime_and_multiplier(current_volatility)
            
            print(f"Current volatility regime: {regime} (volatility: {current_volatility:.6f}, risk multiplier: {risk_multiplier:.2f})")
            
            # Store volatility information in dataframe using proper pandas assignment
            dataframe.loc[:, 'volatility'] = current_volatility
            dataframe.loc[:, 'volatility_regime'] = regime
            dataframe.loc[:, 'risk_multiplier'] = risk_multiplier
            
        else:
            # Insufficient data for volatility calculation
            dataframe.loc[:, 'volatility'] = np.nan
            dataframe.loc[:, 'volatility_regime'] = 'insufficient_data'
            dataframe.loc[:, 'risk_multiplier'] = 1.0
            print(f"Insufficient data for volatility calculation. Need at least 100 candles, got {len(dataframe)}")

        # Initialize all required dataframe columns
        self._initialize_dataframe_columns(dataframe)

        # === Rolling Monte Carlo Optimization Logic (Eliminates Lookahead Bias) ===
        recalc_interval_candles = self.mc_recalc_interval_minutes.value // timeframe_to_minutes(self.timeframe)
        startup_candles = self.mc_lookback_window_candles.value
        min_required_candles = max(self.MIN_LOOKBACK_PERIOD, 100)  # Use minimum required data instead of large startup window
        
        # Check if rolling optimization is enabled
        if not self.enable_rolling_mc_optimization.value:
            print("=== Using Original Monte Carlo Optimization (with potential lookahead bias) ===")
            # Execute Monte Carlo optimization using the manager (original method)
            mc_results = self.monte_carlo_manager.execute_monte_carlo_optimization(
                dataframe, metadata['pair'], self.enable_mc_optimization.value
            )
            
            # Apply the results to the dataframe
            self.monte_carlo_manager.apply_monte_carlo_results(dataframe, mc_results)
            
            # Process convergence analysis if we have results
            if mc_results:
                self._process_convergence_analysis_original(
                    dataframe, 
                    mc_results.get('resistance_score', 0.0), 
                    mc_results.get('support_score', 0.0)
                )
                
                print(f"Applied Monte Carlo results for {metadata['pair']}:")
                print(f"  Resistance Score: {mc_results.get('resistance_score', 0.0):.6f}")
                print(f"  Support Score: {mc_results.get('support_score', 0.0):.6f}")
                print(f"  Optimal Periods: R={mc_results.get('optimal_resistance_period', 0)}, S={mc_results.get('optimal_support_period', 0)}")
            else:
                print(f"Monte Carlo results not yet available for {metadata['pair']}.")
                # Initialize with defaults if no results are available using proper pandas assignment
                dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
                dataframe.loc[:, 'MC_Support_Score'] = 0.0
                dataframe.loc[:, 'score_convergence_ratio'] = 0.0
                dataframe.loc[:, 'convergence_multiplier'] = 1.0
                dataframe.loc[:, 'trading_mode_indicator'] = 0.0
                dataframe.loc[:, 'trading_mode'] = "INSUFFICIENT_DATA"
        
        # Check if we have minimum required data for rolling optimization
        elif len(dataframe) <= min_required_candles:
            print(f"Not enough data for rolling Monte Carlo. Need at least {min_required_candles} candles, got {len(dataframe)}. Skipping Monte Carlo optimization.")
            # Initialize with default values
            dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
            dataframe.loc[:, 'MC_Support_Score'] = 0.0
            dataframe.loc[:, 'score_convergence_ratio'] = 0.0
            dataframe.loc[:, 'convergence_multiplier'] = 1.0
            dataframe.loc[:, 'trading_mode_indicator'] = 0.0
            dataframe.loc[:, 'trading_mode'] = "INSUFFICIENT_DATA"
        else:
            print("=== Using Rolling Monte Carlo Optimization (eliminates lookahead bias) ===")
            last_mc_results = None
            print(f"=== Starting Rolling Monte Carlo Analysis for {metadata['pair']} ===")
            print(f"Recalc interval: {recalc_interval_candles} candles ({self.mc_recalc_interval_minutes.value} minutes)")
            print(f"Lookback window: {self.mc_lookback_window_candles.value} candles")
            print(f"Processing ALL {len(dataframe) - min_required_candles} candles starting from minimum required data ({min_required_candles})")

            # --- State variables for projecting trendlines with slopes ---
            last_resistance_value = np.nan
            last_support_value = np.nan
            current_resistance_slope = 0.0
            current_support_slope = 0.0

            # Start rolling optimization from minimum required data, not startup_candles
            for i in range(min_required_candles, len(dataframe)):
                # Determine if it is time to recalculate
                should_recalculate = (i == min_required_candles) or ((i - min_required_candles) % recalc_interval_candles == 0)

                if should_recalculate:
                    print(f"Recalculating MC results at candle {i}/{len(dataframe)} ({(i/len(dataframe)*100):.1f}%)")
                    # Define the lookback window for the current candle (point-in-time data only)
                    # Use the smaller of: lookback window or all available data up to this point
                    lookback_start = max(0, i - self.mc_lookback_window_candles.value)
                    current_dataframe_slice = dataframe.iloc[lookback_start:i].copy()

                    print(f"  Using data slice: {lookback_start} to {i} ({len(current_dataframe_slice)} candles)")

                    # Execute MC optimization on the slice of data available at this point in time
                    # Create a temporary manager to ensure no state from future data is used
                    temp_mc_manager = MonteCarloManager(
                        mc_iterations=self.MC_ITERATIONS,
                        min_lookback_period=self.MIN_LOOKBACK_PERIOD,
                        recalc_interval_minutes=0,  # Force recalc for temporary manager
                        trendline_proximity_threshold=self.trendline_proximity_threshold.value,
                        trend_analyzer=self.trend_analyzer,
                        volatility_model=self.volatility_model,
                        garch_model=self.garch_model
                    )
                    
                    mc_results = temp_mc_manager.execute_monte_carlo_optimization(
                        current_dataframe_slice, metadata['pair'], self.enable_mc_optimization.value
                    )
                    
                    if mc_results:
                        last_mc_results = mc_results
                        
                        # Update slopes from the new MC results
                        current_resistance_slope = mc_results.get('best_resistance_slope', 0.0)
                        current_support_slope = mc_results.get('best_support_slope', 0.0)
                        
                        # Get the last valid value from the calculated lines as the starting point
                        res_line = mc_results.get('resistance_line', np.array([]))
                        valid_res = res_line[~np.isnan(res_line)]
                        if len(valid_res) > 0:
                            last_resistance_value = valid_res[-1]

                        sup_line = mc_results.get('support_line', np.array([]))
                        valid_sup = sup_line[~np.isnan(sup_line)]
                        if len(valid_sup) > 0:
                            last_support_value = valid_sup[-1]
                        
                        print(f"  New MC results: R_score={mc_results.get('resistance_score', 0):.4f}, S_score={mc_results.get('support_score', 0):.4f}")
                        print(f"  Slopes: R_slope={current_resistance_slope:.6f}, S_slope={current_support_slope:.6f}")

                # Project the trendlines forward using the slope (creating continuous sloped lines)
                pandas_index = dataframe.index[i]
                
                if not np.isnan(last_resistance_value):
                    # Project resistance line forward by adding slope
                    last_resistance_value += current_resistance_slope
                    dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = last_resistance_value
                else:
                    dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = np.nan
                
                if not np.isnan(last_support_value):
                    # Project support line forward by adding slope
                    last_support_value += current_support_slope
                    dataframe.loc[pandas_index, 'MC_Optimal_Support'] = last_support_value
                else:
                    dataframe.loc[pandas_index, 'MC_Optimal_Support'] = np.nan
                
                # Apply scores and convergence analysis
                if last_mc_results:
                    # Apply scores to this specific row
                    dataframe.loc[pandas_index, 'MC_Resistance_Score'] = last_mc_results.get('resistance_score', 0.0)
                    dataframe.loc[pandas_index, 'MC_Support_Score'] = last_mc_results.get('support_score', 0.0)
                    dataframe.loc[pandas_index, 'MC_Optimal_Period'] = last_mc_results.get('optimal_resistance_period', 0.0)
                    
                    # Process convergence analysis for this row
                    self._process_convergence_analysis_for_row(
                        dataframe, i,
                        last_mc_results.get('resistance_score', 0.0),
                        last_mc_results.get('support_score', 0.0)
                    )
                else:
                    # No MC results available yet - initialize with defaults
                    dataframe.loc[pandas_index, 'MC_Resistance_Score'] = 0.0
                    dataframe.loc[pandas_index, 'MC_Support_Score'] = 0.0
                    dataframe.loc[pandas_index, 'MC_Optimal_Period'] = 0.0
                    dataframe.loc[pandas_index, 'score_convergence_ratio'] = 0.0
                    dataframe.loc[pandas_index, 'convergence_multiplier'] = 1.0
                    dataframe.loc[pandas_index, 'trading_mode_indicator'] = 0.0
                    dataframe.loc[pandas_index, 'trading_mode'] = "INSUFFICIENT_DATA"

            print(f"Rolling Monte Carlo optimization completed for {metadata['pair']}")
            if last_mc_results:
                print(f"Final results: R_score={last_mc_results.get('resistance_score', 0):.4f}, S_score={last_mc_results.get('support_score', 0):.4f}")
                print(f"Final slopes: R_slope={current_resistance_slope:.6f}, S_slope={current_support_slope:.6f}")
                
            # Initialize the early candles that couldn't be processed with rolling optimization
            for i in range(0, min_required_candles):
                pandas_index = dataframe.index[i]
                dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = np.nan
                dataframe.loc[pandas_index, 'MC_Optimal_Support'] = np.nan
                dataframe.loc[pandas_index, 'MC_Resistance_Score'] = 0.0
                dataframe.loc[pandas_index, 'MC_Support_Score'] = 0.0
                dataframe.loc[pandas_index, 'MC_Optimal_Period'] = 0.0
                dataframe.loc[pandas_index, 'score_convergence_ratio'] = 0.0
                dataframe.loc[pandas_index, 'convergence_multiplier'] = 1.0
                dataframe.loc[pandas_index, 'trading_mode_indicator'] = 0.0
                dataframe.loc[pandas_index, 'trading_mode'] = "INSUFFICIENT_DATA"

        # Calculate linear regression trendlines using TA-Lib
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
            
            # Copy the linear regression columns back to the original dataframe using proper pandas assignment
            dataframe.loc[:, 'linear_reg'] = linearreg_data['linear_reg']
            dataframe.loc[:, 'linear_reg_slope'] = linearreg_data['linear_reg_slope']
            dataframe.loc[:, 'linear_reg_angle'] = linearreg_data['linear_reg_angle']
            dataframe.loc[:, 'linear_reg_intercept'] = linearreg_data['linear_reg_intercept']
            dataframe.loc[:, 'linear_reg_line'] = linearreg_data['linear_reg_line']
                    
        except Exception as e:
            print(f"Error calculating linear regression: {e}")

        # Mark high and low points for visualization
        try:
            # Find and map swing points to both dataframes using TrendAnalysis method
            self.trend_analyzer.find_and_map_swing_points(dataframe, dataframe)
            
        except Exception as e:
            print(f"Error finding swing points: {e}")

        # === Calculate and Store Bounce Metrics for Chart Display ===
        try:
            print("=== Calculating Bounce Metrics for Chart Display ===")
            bounce_metrics = self.calculate_bounce_metrics(dataframe)
            
            # Store bounce counts as constant values across all rows for chart display
            dataframe.loc[:, 'resistance_bounce_count'] = bounce_metrics['resistance_bounce_count']
            dataframe.loc[:, 'support_bounce_count'] = bounce_metrics['support_bounce_count']
            
            # Store normalized scores for better chart visualization (divide by 100 to fit with bounce counts)
            dataframe.loc[:, 'resistance_bounce_score_display'] = bounce_metrics['resistance_score'] / 100.0
            dataframe.loc[:, 'support_bounce_score_display'] = bounce_metrics['support_score'] / 100.0
            
            print(f"Bounce Metrics Summary:")
            print(f"  Resistance: {bounce_metrics['resistance_bounce_count']:.0f} bounces, score: {bounce_metrics['resistance_score']:.4f}")
            print(f"  Support: {bounce_metrics['support_bounce_count']:.0f} bounces, score: {bounce_metrics['support_score']:.4f}")
            
        except Exception as e:
            print(f"Error calculating bounce metrics: {e}")
            # Initialize with default values if calculation fails
            dataframe.loc[:, 'resistance_bounce_count'] = 0.0
            dataframe.loc[:, 'support_bounce_count'] = 0.0
            dataframe.loc[:, 'resistance_bounce_score_display'] = 0.0
            dataframe.loc[:, 'support_bounce_score_display'] = 0.0

        
        # Uncomment the line below to run GARCH examples for demonstration
        # from risk_metrics.volatility_models import run_garch_examples
        # run_garch_examples()
            
        return dataframe

    def _process_convergence_analysis_for_row(self, dataframe: DataFrame, index: int,
                                              raw_resistance_score: float,
                                              raw_support_score: float) -> None:
        """
        Process volatility regime and convergence analysis, storing results in a specific row.
        
        Args:
            dataframe: DataFrame to store analysis results
            index: The row index to store results in
            raw_resistance_score: Raw resistance score from Monte Carlo
            raw_support_score: Raw support score from Monte Carlo
        """
        convergence_ratio = self.calculate_score_convergence_ratio(raw_support_score, raw_resistance_score)
        convergence_multiplier = self.get_convergence_multiplier(convergence_ratio)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        trading_mode_mapping = {
            "NORMAL_BOUNCE": 1.0, "LOW_CONVERGENCE": 2.0,
            "MEDIUM_CONVERGENCE": 3.0, "HIGH_CONVERGENCE": 4.0
        }
        trading_mode_indicator = trading_mode_mapping.get(trading_mode, 0.0)
        
        pandas_index = dataframe.index[index]
        dataframe.loc[pandas_index, 'score_convergence_ratio'] = convergence_ratio
        dataframe.loc[pandas_index, 'convergence_multiplier'] = convergence_multiplier
        dataframe.loc[pandas_index, 'trading_mode_indicator'] = trading_mode_indicator
        dataframe.loc[pandas_index, 'trading_mode'] = trading_mode

    def _process_convergence_analysis_original(self, dataframe: DataFrame, 
                                             raw_resistance_score: float, 
                                             raw_support_score: float) -> None:
        """
        Process volatility regime and convergence analysis, storing results in dataframe.
        This is the original method used when rolling optimization is disabled.
        
        Args:
            dataframe: DataFrame to store analysis results
            raw_resistance_score: Raw resistance score from Monte Carlo
            raw_support_score: Raw support score from Monte Carlo
        """
        print("=== Trendlines Score Convergence Analysis (Original Method) ===")
        
        # Calculate convergence ratio between MC scores
        convergence_ratio = self.calculate_score_convergence_ratio(raw_support_score, raw_resistance_score)
        convergence_multiplier = self.get_convergence_multiplier(convergence_ratio)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        # Convert trading mode to numeric indicator for plotting
        trading_mode_mapping = {
            "NORMAL_BOUNCE": 1.0,
            "LOW_CONVERGENCE": 2.0,
            "MEDIUM_CONVERGENCE": 3.0,
            "HIGH_CONVERGENCE": 4.0
        }
        trading_mode_indicator = trading_mode_mapping.get(trading_mode, 0.0)
        
        # Store convergence metrics in dataframe
        dataframe.loc[:, 'score_convergence_ratio'] = convergence_ratio
        dataframe.loc[:, 'convergence_multiplier'] = convergence_multiplier
        dataframe.loc[:, 'trading_mode_indicator'] = trading_mode_indicator
        dataframe.loc[:, 'trading_mode'] = trading_mode
        
        # Log convergence analysis results
        print(f"Score Convergence Analysis Results:")
        print(f"  Support Score: {raw_support_score:.6f}")
        print(f"  Resistance Score: {raw_resistance_score:.6f}")
        print(f"  Convergence Ratio: {convergence_ratio:.3f}")
        print(f"  Convergence Multiplier: {convergence_multiplier:.3f}")
        print(f"  Trading Mode: {trading_mode}")
        print(f"  Trading Mode Indicator: {trading_mode_indicator}")
        
        # Get detailed breakout scenario analysis
        breakout_analysis = self.detect_breakout_scenario(dataframe)
        print(f"  Breakout Analysis: {breakout_analysis['recommendation']}")

    def _initialize_dataframe_columns(self, dataframe: DataFrame) -> None:
        """
        Initialize all required columns in the dataframe with default values.
        
        Args:
            dataframe: DataFrame to initialize columns in
        """
        # Initialize marker columns for support and resistance points using proper pandas assignment
        dataframe.loc[:, 'all_highs'] = np.nan
        dataframe.loc[:, 'all_lows'] = np.nan
        
        # Initialize columns for main trend lines
        dataframe.loc[:, 'Resistance Line'] = np.nan
        dataframe.loc[:, 'Support Line'] = np.nan
        
        # Initialize Monte Carlo optimal lines
        dataframe.loc[:, 'MC_Optimal_Resistance'] = np.nan
        dataframe.loc[:, 'MC_Optimal_Support'] = np.nan
        dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
        dataframe.loc[:, 'MC_Support_Score'] = 0.0
        dataframe.loc[:, 'MC_Optimal_Period'] = 0.0
        
        # Initialize columns for highest scored line
        dataframe.loc[:, 'Highest_Scored_Line'] = np.nan
        dataframe.loc[:, 'Highest_Set_Mean'] = np.nan
        dataframe.loc[:, 'Highest_Line_Score'] = np.nan
        dataframe.loc[:, 'Highest_Line_Type'] = ""  # Will be "Resistance" or "Support"
        dataframe.loc[:, 'Highest_Line_Text'] = ""  # For displaying text on the chart
        
        # Initialize columns for individual segment trend lines
        max_segments = 10
        for i in range(max_segments):
            dataframe.loc[:, f'Max_Line_{i}'] = np.nan
            dataframe.loc[:, f'Min_Line_{i}'] = np.nan

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals using the modular SignalGenerator.
        Bounce Trading Strategy Implementation using Price Extrema with Convergence Detection.
        Now uses historically accurate rolling Monte Carlo calculations - no lookahead bias.
        """
        return self.signal_generator.generate_entry_signals(dataframe)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals using the modular SignalGenerator.
        Enhanced exit signals for bounce trading with cross-signal exits.
        Now uses historically accurate rolling Monte Carlo calculations - no lookahead bias.
        """
        return self.signal_generator.generate_exit_signals(dataframe)


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

