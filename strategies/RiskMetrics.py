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
from risk_metrics.monte_carlo import MonteCarloSimulator
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
            dataframe['close'], dataframe['high'], dataframe['low'], 
            dataframe['MC_Optimal_Support'], 'long', self.strategy.trendline_proximity_threshold.value
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
            dataframe['close'], dataframe['high'], dataframe['low'], 
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy.trendline_proximity_threshold.value
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
            dataframe['close'], dataframe['high'], dataframe['low'], 
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy.trendline_proximity_threshold.value
        )
        
        # Generate long entry conditions for short exits - using direct import from trendline.py
        long_entry_conditions = generate_bounce_conditions(
            dataframe['close'], dataframe['high'], dataframe['low'], 
            dataframe['MC_Optimal_Support'], 'long', self.strategy.trendline_proximity_threshold.value
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
    
    # Monte Carlo period optimization settings
    MC_ITERATIONS = 1000
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
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.002, space="buy", optimize=True)
    
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
    enable_convergence_detection = BooleanParameter(default=True, space="buy", optimize=False)
    
    # === ATR-based Custom Stoploss Parameters ===
    use_custom_stoploss = True
    stoploss_atr_multiplier = DecimalParameter(1.0, 5.0, default=3.0, space="buy", optimize=True)
    # Additional stoploss safety parameters
    stoploss_min_percent = DecimalParameter(0.005, 0.02, default=0.01, space="buy", optimize=True)  # Minimum 0.5-2% stoploss
    stoploss_max_percent = DecimalParameter(0.08, 0.25, default=0.15, space="buy", optimize=True)  # Maximum 8-25% stoploss
    stoploss_profit_protection = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)  # Tighten stoploss after 1-5% profit
    
    # === New Periodic Monte Carlo Parameters ===
    mc_recalc_interval_minutes = IntParameter(60, 480, default=240, space="buy", optimize=False)
    mc_lookback_window_candles = IntParameter(1000, 3000, default=2000, space="buy", optimize=False)

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
        
        # === New State Variables for Periodic MC ===
        self.last_mc_recalc_time: Optional[datetime] = None
        self.cached_mc_results: Dict[str, any] = {}
        
        print(f"RiskMetrics strategy initialized with separated GARCH implementation:")
        print(f"  GARCH model: Used for risk management and volatility estimation only")
        print(f"  Lookback periods: Fixed periods for Monte Carlo optimization")
        print(f"  Monte Carlo iterations: {self.MC_ITERATIONS}")
        print(f"  Score convergence detection enabled: {self.enable_convergence_detection.value}")
        print(f"  Convergence thresholds: High={self.score_convergence_high_threshold.value:.2f}, Medium={self.score_convergence_medium_threshold.value:.2f}, Low={self.score_convergence_low_threshold.value:.2f}")
        print(f"  Proper GARCH usage: Volatility forecasting separate from technical analysis optimization")
        print(f"  Signal generator: Initialized for modular signal generation")
        print(f"  Periodic Monte Carlo: Recalc interval={self.mc_recalc_interval_minutes.value} minutes, Lookback window={self.mc_lookback_window_candles.value} candles")
        print(f"  Enhanced ATR Stoploss: Enabled with {self.stoploss_atr_multiplier.value:.1f}x ATR multiplier")
        print(f"    - Stoploss bounds: Min={self.stoploss_min_percent.value*100:.1f}%, Max={self.stoploss_max_percent.value*100:.1f}%")
        print(f"    - Profit protection: Activates at {self.stoploss_profit_protection.value*100:.1f}% profit")

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

    def generate_fixed_lookback_periods(self, dataframe: DataFrame) -> List[int]:
        """
        Generate fixed lookback periods for Monte Carlo optimization.
        This replaces the volatility-based period generation to properly separate 
        GARCH volatility estimation from lookback period selection.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            List[int]: List of fixed lookback periods for testing
        """
        total_candles = len(dataframe)
        max_lookback_period = total_candles  # Use all available data
        min_lookback_period = max(self.MIN_LOOKBACK_PERIOD, 50)  # Use fixed min
        
        # Define core fixed periods that cover different time horizons
        core_periods = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # Filter periods based on available data
        valid_core_periods = [p for p in core_periods if min_lookback_period <= p <= max_lookback_period]
        
        # Generate additional random periods to reach MC_ITERATIONS
        remaining_iterations = self.MC_ITERATIONS - len(valid_core_periods)
        random_periods = []
        
        if remaining_iterations > 0:
            # Generate random periods to fill the remaining iterations
            import random
            for _ in range(remaining_iterations):
                random_period = random.randint(min_lookback_period, max_lookback_period)
                random_periods.append(random_period)
        
        # Combine core periods with random periods
        all_periods = valid_core_periods + random_periods
        
        # Ensure we have exactly MC_ITERATIONS periods
        if len(all_periods) > self.MC_ITERATIONS:
            all_periods = all_periods[:self.MC_ITERATIONS]
        elif len(all_periods) < self.MC_ITERATIONS:
            # Pad with repeated core periods if needed
            while len(all_periods) < self.MC_ITERATIONS:
                all_periods.extend(valid_core_periods[:self.MC_ITERATIONS - len(all_periods)])
        
        # Shuffle to randomize the order
        random.shuffle(all_periods)
        
        print(f"Generated {len(all_periods)} fixed lookback periods:")
        print(f"  Range: {min(all_periods)} to {max(all_periods)} candles")
        print(f"  Mean: {np.mean(all_periods):.1f}, Std: {np.std(all_periods):.1f}")
        print(f"  Core periods included: {valid_core_periods}")
        
        return all_periods

    def estimate_current_volatility_regime(self, dataframe: DataFrame) -> Tuple[str, float, float]:
        """
        Estimate current volatility regime using GARCH model for risk management purposes.
        This is separated from lookback period generation to ensure proper use of GARCH.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Tuple[str, float, float]: (regime, volatility, risk_multiplier)
        """
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
        
        return regime, current_volatility, risk_multiplier


    def monte_carlo_period_optimization(self, dataframe: DataFrame) -> Dict[str, any]:
        """
        Use Monte Carlo simulation to test different lookback periods and find
        the ones that produce the highest scoring resistance and support lines.
        Now uses fixed lookback periods
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Dict containing optimal periods and their scores
        """
        # Set MAX_LOOKBACK_PERIOD dynamically based on available data
        total_candles = len(dataframe)
        max_lookback_period = min(total_candles, 1000)  # Fixed max instead of garch_max_candles
        
        if total_candles < self.MIN_LOOKBACK_PERIOD:
            print(f"Not enough data for Monte Carlo optimization. Need at least {self.MIN_LOOKBACK_PERIOD} candles, got {total_candles}.")
            return {
                'optimal_resistance_period': self.MIN_LOOKBACK_PERIOD,
                'optimal_support_period': self.MIN_LOOKBACK_PERIOD,
                'resistance_score': 0.0,
                'support_score': 0.0,
                'resistance_line': np.full(len(dataframe), np.nan),
                'support_line': np.full(len(dataframe), np.nan)
            }
        
        # Seed random number generator for reproducible results with some variability
        import time
        random.seed(int(time.time() * 1000) % 10000)  # Use current time for seed
        
        # Generate fixed lookback periods
        print("=== Monte Carlo Period Optimization with Fixed Periods ===")
        lookback_periods = self.generate_fixed_lookback_periods(dataframe)
        
        # Separately estimate volatility regime for risk management
        regime, current_volatility, risk_multiplier = self.estimate_current_volatility_regime(dataframe)
        
        print(f"Starting Monte Carlo period optimization with {len(lookback_periods)} iterations...")
        print(f"Testing periods from {min(lookback_periods)} to {max(lookback_periods)} candles (total data: {total_candles})")
        print(f"Volatility regime for risk management: {regime} (volatility: {current_volatility:.6f})")
        
        best_resistance_score = 0.0
        best_support_score = 0.0
        best_resistance_period = self.MIN_LOOKBACK_PERIOD
        best_support_period = self.MIN_LOOKBACK_PERIOD
        best_resistance_line = np.full(len(dataframe), np.nan)
        best_support_line = np.full(len(dataframe), np.nan)
        
        # Store all tested periods and scores for analysis
        tested_periods = []
        resistance_scores = []
        support_scores = []
        
        # Track period distribution for debugging
        period_counts = {}
        
        for iteration, random_period in enumerate(lookback_periods):
            # Track period distribution
            period_range = (random_period // 100) * 100  # Group by hundreds
            period_counts[period_range] = period_counts.get(period_range, 0) + 1
            
            try:
                # Test this period
                recent_data = dataframe.tail(random_period).copy()
                
                # Use the new helper method to generate trendlines and scores
                trendline_results = self._generate_trendlines_for_period(recent_data, random_period)
                
                current_resistance_score = trendline_results['resistance_score']
                current_support_score = trendline_results['support_score']
                trends = trendline_results['trends']
                
                # Store results
                tested_periods.append(random_period)
                resistance_scores.append(current_resistance_score)
                support_scores.append(current_support_score)
                
                # Check if this is the best resistance line so far
                if current_resistance_score > best_resistance_score:
                    best_resistance_score = current_resistance_score
                    best_resistance_period = random_period
                    
                    # Store the resistance line
                    resistance_line = np.full(len(dataframe), np.nan)
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            resistance_line[current_idx] = trends['Max Line'].iloc[i]
                    best_resistance_line = resistance_line
                    
                    print(f"New best resistance found at iteration {iteration + 1}: period {random_period}, score {current_resistance_score:.4f}")
                
                # Check if this is the best support line so far
                if current_support_score > best_support_score:
                    best_support_score = current_support_score
                    best_support_period = random_period
                    
                    # Store the support line
                    support_line = np.full(len(dataframe), np.nan)
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            support_line[current_idx] = trends['Min Line'].iloc[i]
                    best_support_line = support_line
                    
                    print(f"New best support found at iteration {iteration + 1}: period {random_period}, score {current_support_score:.4f}")
                
                # Progress reporting with more details
                if (iteration + 1) % 100 == 0:
                    print(f"Monte Carlo progress: {iteration + 1}/{len(lookback_periods)} iterations completed")
                    print(f"Current best - Resistance: {best_resistance_score:.4f} (period {best_resistance_period}), Support: {best_support_score:.4f} (period {best_support_period})")
                    print(f"Last 5 tested periods: {tested_periods[-5:] if len(tested_periods) >= 5 else tested_periods}")
                
            except Exception as e:
                print(f"Error in Monte Carlo iteration {iteration}: {e}")
                continue
        
        print(f"Fixed-period Monte Carlo optimization completed!")
        print(f"Optimal resistance period: {best_resistance_period} (score: {best_resistance_score:.4f})")
        print(f"Optimal support period: {best_support_period} (score: {best_support_score:.4f})")
        print(f"Tested periods range: {min(tested_periods) if tested_periods else 'N/A'} to {max(tested_periods) if tested_periods else 'N/A'} candles")
        
        # Print period distribution for debugging
        print("Period distribution (grouped by hundreds):")
        for period_range in sorted(period_counts.keys()):
            print(f"  {period_range}-{period_range+99}: {period_counts[period_range]} times")
        
        # Print some statistics about the tested periods
        if tested_periods:
            print(f"Period statistics - Min: {min(tested_periods)}, Max: {max(tested_periods)}, Mean: {np.mean(tested_periods):.1f}")
            print(f"Unique periods tested: {len(set(tested_periods))} out of {len(tested_periods)} total")
        
        return {
            'optimal_resistance_period': best_resistance_period,
            'optimal_support_period': best_support_period,
            'resistance_score': best_resistance_score,
            'support_score': best_support_score,
            'resistance_line': best_resistance_line,
            'support_line': best_support_line,
            'tested_periods': tested_periods,
            'all_resistance_scores': resistance_scores,
            'all_support_scores': support_scores,
            'max_lookback_period': max_lookback_period,
            'period_distribution': period_counts,
            'volatility_regime': regime,
            'current_volatility': current_volatility,
            'risk_multiplier': risk_multiplier
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

        # Calculate and store volatility regime information using separated GARCH estimation
        if len(dataframe) >= 100:  # Use fixed minimum threshold for volatility estimation
            print("=== GARCH Volatility Estimation for Risk Management ===")
            regime, current_volatility, risk_multiplier = self.estimate_current_volatility_regime(dataframe)
            
            # Store volatility information in dataframe
            dataframe['volatility'] = current_volatility
            dataframe['volatility_regime'] = regime
            dataframe['risk_multiplier'] = risk_multiplier
            
        else:
            # Insufficient data for volatility calculation
            dataframe['volatility'] = np.nan
            dataframe['volatility_regime'] = 'insufficient_data'
            dataframe['risk_multiplier'] = 1.0
            print(f"Insufficient data for volatility calculation. Need at least 100 candles, got {len(dataframe)}")

        # Initialize all required dataframe columns
        self._initialize_dataframe_columns(dataframe)

        # === Periodic Monte Carlo Optimization Logic ===
        
        # Condition to trigger a full recalculation
        should_recalc_mc = False
        if self.enable_mc_optimization.value:
            # Get current time - use 'date' column if available, otherwise use current time
            if 'date' in dataframe.columns:
                current_time = dataframe['date'].iloc[-1]
                if not isinstance(current_time, datetime):
                    current_time = pd.to_datetime(current_time)
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
            else:
                current_time = datetime.now(timezone.utc)
            
            if self.last_mc_recalc_time is None:
                should_recalc_mc = True
                print("First run: Initializing Monte Carlo optimization.")
            else:
                time_since_last_recalc = (current_time - self.last_mc_recalc_time).total_seconds() / 60
                if time_since_last_recalc >= self.mc_recalc_interval_minutes.value:
                    should_recalc_mc = True
                    print(f"Time to recalculate MC: {time_since_last_recalc:.1f} minutes elapsed (threshold: {self.mc_recalc_interval_minutes.value})")

        # --- HEAVY CALCULATION BLOCK ---
        if should_recalc_mc:
            self.last_mc_recalc_time = current_time
            
            # Calculate MAX_LOOKBACK_PERIOD dynamically based on available data (same as in monte_carlo_period_optimization)
            total_candles = len(dataframe)
            max_lookback_period = min(total_candles, 1000)  # Use the same logic as in the MC method
            
            # Use the dynamically calculated lookback window
            lookback_window = min(self.mc_lookback_window_candles.value, max_lookback_period)
            
            if len(dataframe) >= lookback_window:
                print(f"Running MC optimization on last {lookback_window} candles (max available: {max_lookback_period}).")
                mc_data = dataframe.tail(lookback_window).copy()
                
                # Run the full optimization and cache the results
                self.cached_mc_results = self.monte_carlo_period_optimization(mc_data)
            else:
                print(f"Not enough data for MC optimization ({len(dataframe)} < {lookback_window}).")

        # --- LIGHTWEIGHT APPLICATION BLOCK (runs every time) ---
        if self.cached_mc_results:
            # Apply cached global scores and optimal periods
            dataframe['MC_Resistance_Score'] = self.cached_mc_results['resistance_score']
            dataframe['MC_Support_Score'] = self.cached_mc_results['support_score']
            dataframe['MC_Optimal_Period'] = max(
                self.cached_mc_results['optimal_resistance_period'],
                self.cached_mc_results['optimal_support_period']
            )

            # Apply the optimal lines - these were calculated on a window, so we apply them to the tail
            resistance_line = self.cached_mc_results['resistance_line']
            support_line = self.cached_mc_results['support_line']
            
            # Align the calculated lines with the main dataframe
            if len(resistance_line) > 0 and not np.isnan(resistance_line).all():
                start_idx = max(0, len(dataframe) - len(resistance_line))
                end_idx = len(dataframe)
                dataframe.iloc[start_idx:end_idx, dataframe.columns.get_loc('MC_Optimal_Resistance')] = resistance_line[-len(dataframe[start_idx:end_idx]):]
            if len(support_line) > 0 and not np.isnan(support_line).all():
                start_idx = max(0, len(dataframe) - len(support_line))
                end_idx = len(dataframe)
                dataframe.iloc[start_idx:end_idx, dataframe.columns.get_loc('MC_Optimal_Support')] = support_line[-len(dataframe[start_idx:end_idx]):]

            # Process convergence analysis using the globally optimal scores
            # This will still produce a single value, which is correct for this architecture
            self._process_convergence_analysis(
                dataframe, 
                self.cached_mc_results['resistance_score'], 
                self.cached_mc_results['support_score']
            )
            
            print(f"Applied cached Monte Carlo results:")
            print(f"  Resistance Score: {self.cached_mc_results['resistance_score']:.6f}")
            print(f"  Support Score: {self.cached_mc_results['support_score']:.6f}")
            print(f"  Optimal Periods: R={self.cached_mc_results['optimal_resistance_period']}, S={self.cached_mc_results['optimal_support_period']}")
            
        else:
            print("Monte Carlo results not yet available.")
            # Initialize with defaults if no results are cached
            dataframe['MC_Resistance_Score'] = 0.0
            dataframe['MC_Support_Score'] = 0.0
            dataframe['score_convergence_ratio'] = 0.0
            dataframe['convergence_multiplier'] = 1.0
            dataframe['trading_mode_indicator'] = 0.0
            dataframe['trading_mode'] = "INSUFFICIENT_DATA"

        # Calculate trendlines using local maxima/minima
        lookback = 200  # Use last 200 candles for trendline calculation

        # Don't calculate trendlines if we don't have enough data
        if len(dataframe) < 30:
            return dataframe
            
        recent_data = dataframe.tail(lookback).copy()
        
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
            # Find and map swing points to both dataframes
            high_swing_points, low_swing_points = self._find_and_map_swing_points(dataframe, recent_data)
            
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
                    dataframe.loc[current_idx, 'Resistance Line'] = trends['Max Line'].iloc[i]
                    dataframe.loc[current_idx, 'Support Line'] = trends['Min Line'].iloc[i]
        except Exception as e:
            # Fail gracefully if gentrends fails
            print(f"Error in gentrends: {e}")
        
        # Use segtrends with different segment counts to find multiple resistance and support lines
        try:
            # Generate segmented trends
            max_segments = 10
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
        
        # Calculate and store trendline scores
        self._calculate_and_store_trendline_scores(dataframe, recent_data, trends, seg_trends)

        # GARCH Examples and Risk Calculations
        self._run_garch_examples()
            
        return dataframe

    def _run_garch_examples(self) -> None:
        """
        Run GARCH examples and risk calculations for demonstration purposes.
        """
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

        # Example 2: Monte Carlo simulation with GARCH using the new MonteCarloSimulator
        monte_carlo = MonteCarloSimulator(garch_model)
        simulated_returns = monte_carlo.simulate_with_garch(T=3, iterations=1000)
        confidence_level = 0.01
        VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
            simulated_returns=simulated_returns,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(simulated_returns)
        )
        print(f"VaR à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
        print(f"ES à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")
        
        # Example 3: Monte Carlo with custom distribution (Student's t)
        print("--------------------------------")
        print("Begin Monte Carlo with Student's t-distribution")        
        print("---")
        
        simulated_returns_t = monte_carlo.simulate_with_custom_distribution(T=3, iterations=1000)
        VaR_1_percent_t, ES_1_percent_t = risk_indicators.calculate_var_es(
            simulated_returns=simulated_returns_t,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(simulated_returns_t)
        )
        print(f"VaR à 1% sur 3 jours avec distribution t de Student: {VaR_1_percent_t:.4f} ({VaR_1_percent_t * 100:.2f}%)")
        print(f"ES à 1% sur 3 jours avec distribution t de Student: {ES_1_percent_t:.4f} ({ES_1_percent_t * 100:.2f}%)")
        
        # Display simulation statistics
        stats = monte_carlo.get_simulation_statistics(simulated_returns)
        print(f"Statistiques de simulation - Moyenne: {stats['mean']:.4f}, Écart-type: {stats['std']:.4f}")
        print(f"Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")

    def _find_and_map_swing_points(self, dataframe: DataFrame, recent_data: DataFrame) -> Tuple[List, List]:
        """
        Find swing points and map them to both dataframes.
        
        Args:
            dataframe: Full dataframe to map swing points to
            recent_data: Recent data subset for swing point detection
            
        Returns:
            Tuple[List, List]: (high_swing_points, low_swing_points)
        """
        # Find swing points using TrendAnalysis
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
        
        # Initialize swing point columns in recent_data
        recent_data['all_highs'] = np.nan
        recent_data['all_lows'] = np.nan
        
        # Map swing points to both dataframes
        for idx, price in high_swing_points:
            if idx < len(recent_data):
                actual_idx = len(dataframe) - len(recent_data) + idx
                if 0 <= actual_idx < len(dataframe):
                    dataframe.loc[actual_idx, 'all_highs'] = price
                    recent_data.iloc[idx, recent_data.columns.get_loc('all_highs')] = price
        
        for idx, price in low_swing_points:
            if idx < len(recent_data):
                actual_idx = len(dataframe) - len(recent_data) + idx
                if 0 <= actual_idx < len(dataframe):
                    dataframe.loc[actual_idx, 'all_lows'] = price
                    recent_data.iloc[idx, recent_data.columns.get_loc('all_lows')] = price
        
        return high_swing_points, low_swing_points

    def _generate_trendlines_for_period(self, recent_data: DataFrame, random_period: int) -> Dict[str, any]:
        """
        Generate trendlines and calculate scores for a specific lookback period.
        
        Args:
            recent_data: DataFrame with recent OHLCV data
            random_period: Lookback period to test
            
        Returns:
            Dict containing trendlines and scores
        """
        # Find swing points for this period with adaptive parameters
        high_swing_points = self.trend_analyzer._find_swing_points(
            prices=recent_data['high'].values,
            price_type='high',
            min_points=max(3, random_period // 50),  # Adaptive min_points
            distance=max(5, random_period // 100)    # Adaptive distance
        )
        
        low_swing_points = self.trend_analyzer._find_swing_points(
            prices=recent_data['low'].values,
            price_type='low',
            min_points=max(3, random_period // 50),
            distance=max(5, random_period // 100)
        )
        
        # Initialize swing point columns
        recent_data['all_highs'] = np.nan
        recent_data['all_lows'] = np.nan
        
        # Map swing points
        for idx, price in high_swing_points:
            if idx < len(recent_data):
                recent_data.iloc[idx, recent_data.columns.get_loc('all_highs')] = price
        
        for idx, price in low_swing_points:
            if idx < len(recent_data):
                recent_data.iloc[idx, recent_data.columns.get_loc('all_lows')] = price
        
        # Generate trends for this period
        trends = gentrends(recent_data, field='close', window=1/3.0)
        
        # Calculate scores for the main lines
        main_lines_score = rank_trendlines(
            trends,
            price_field="Data", 
            threshold=self.trendline_proximity_threshold.value,
            all_highs=recent_data['all_highs'],
            all_lows=recent_data['all_lows'],
            pivot_bonus=10.0  # Higher bonus for MC optimization
        )
        
        resistance_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
        support_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
        
        return {
            'trends': trends,
            'resistance_score': resistance_score,
            'support_score': support_score,
            'high_swing_points': high_swing_points,
            'low_swing_points': low_swing_points
        }

    def _calculate_and_store_trendline_scores(self, dataframe: DataFrame, recent_data: DataFrame, 
                                           trends: DataFrame, seg_trends: DataFrame) -> None:
        """
        Calculate trendline scores and store them in the dataframe.
        
        Args:
            dataframe: Full dataframe to store scores in
            recent_data: Recent data with swing points
            trends: Main trendlines from gentrends
            seg_trends: Segmented trendlines from segtrends
        """
        try:
            # Rank trendlines based on proximity to price
            trendline_rankings = rank_trendlines(
                seg_trends, 
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                max_prefix="Max_Line_", 
                min_prefix="Min_Line_",
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=8.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Store the top ranked maxlines and minlines
            ranked_maxlines = trendline_rankings["ranked_maxlines"]
            ranked_minlines = trendline_rankings["ranked_minlines"]
            
            # Calculate mean scores for each set and select the highest scored line
            scores_result = self.trend_analyzer.calculate_trendline_set_scores(ranked_maxlines, ranked_minlines)
            
            # Store mean scores in the dataframe
            resistance_mean_score = scores_result["max_mean_score"]
            support_mean_score = scores_result["min_mean_score"]
            highest_set = scores_result["highest_set"]
            highest_line_name = scores_result["highest_line_name"]
            highest_line_score = scores_result["highest_line_score"]
            
            # Add indicators for visualization
            dataframe['Resistance_Mean_Score'] = resistance_mean_score
            dataframe['Support_Mean_Score'] = support_mean_score
            
            # Calculate and store scores for the main Max Line and Min Line
            main_lines_score = rank_trendlines(
                trends,  # Use the gentrends output that has Resistance Line and Support Line
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=9.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Extract the scores for Resistance Line and Support Line
            resistance_line_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
            support_line_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
            
            # Store the scores in the dataframe
            dataframe['Resistance_Line_Score'] = resistance_line_score
            dataframe['Support_Line_Score'] = support_line_score
            
            # Copy the highest scored line to the Highest_Scored_Line column
            if highest_line_name is not None:
                last_idx = len(dataframe) - len(recent_data)
                for i in range(len(seg_trends)):
                    current_idx = last_idx + i
                    if current_idx < len(dataframe):
                        dataframe.loc[current_idx, 'Highest_Scored_Line'] = seg_trends[highest_line_name].iloc[i]
                        dataframe.loc[current_idx, 'Highest_Set_Mean'] = resistance_mean_score if highest_set == "max" else support_mean_score
                        dataframe.loc[current_idx, 'Highest_Line_Score'] = highest_line_score
                        dataframe.loc[current_idx, 'Highest_Line_Type'] = "Resistance" if highest_set == "max" else "Support"
                        
        except Exception as e:
            print(f"Error in calculating trendline scores: {e}")

    def _initialize_dataframe_columns(self, dataframe: DataFrame) -> None:
        """
        Initialize all required columns in the dataframe with default values.
        
        Args:
            dataframe: DataFrame to initialize columns in
        """
        # Initialize marker columns for support and resistance points
        dataframe['all_highs'] = np.nan
        dataframe['all_lows'] = np.nan
        
        # Initialize columns for main trend lines
        dataframe['Resistance Line'] = np.nan
        dataframe['Support Line'] = np.nan
        
        # Initialize Monte Carlo optimal lines
        dataframe['MC_Optimal_Resistance'] = np.nan
        dataframe['MC_Optimal_Support'] = np.nan
        dataframe['MC_Resistance_Score'] = 0.0
        dataframe['MC_Support_Score'] = 0.0
        dataframe['MC_Optimal_Period'] = 0.0
        
        # Initialize columns for highest scored line
        dataframe['Highest_Scored_Line'] = np.nan
        dataframe['Highest_Set_Mean'] = np.nan
        dataframe['Highest_Line_Score'] = np.nan
        dataframe['Highest_Line_Type'] = ""  # Will be "Resistance" or "Support"
        dataframe['Highest_Line_Text'] = ""  # For displaying text on the chart
        
        # Initialize columns for individual segment trend lines
        max_segments = 10
        for i in range(max_segments):
            dataframe[f'Max_Line_{i}'] = np.nan
            dataframe[f'Min_Line_{i}'] = np.nan

    def _process_convergence_analysis(self, dataframe: DataFrame, 
                                                   raw_resistance_score: float, 
                                                   raw_support_score: float) -> None:
        """
        Process volatility regime and convergence analysis, storing results in dataframe.
        
        Args:
            dataframe: DataFrame to store analysis results
            raw_resistance_score: Raw resistance score from Monte Carlo
            raw_support_score: Raw support score from Monte Carlo
        """
        print("=== Monte Carlo Score Convergence Analysis ===")
        
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
        dataframe['score_convergence_ratio'] = convergence_ratio
        dataframe['convergence_multiplier'] = convergence_multiplier
        dataframe['trading_mode_indicator'] = trading_mode_indicator
        dataframe['trading_mode'] = trading_mode
        
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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals using the modular SignalGenerator.
        Bounce Trading Strategy Implementation using Price Extrema with Convergence Detection.
        """
        return self.signal_generator.generate_entry_signals(dataframe)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals using the modular SignalGenerator.
        Enhanced exit signals for bounce trading with cross-signal exits.
        """
        return self.signal_generator.generate_exit_signals(dataframe)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced custom stoploss based on ATR with dynamic adjustment.
        This function is called for every candle for open trades.
        
        Features:
        - ATR-based dynamic stoploss calculation
        - Profit protection (tighten stoploss when in profit)
        - Volatility regime adjustment
        - Support/resistance level awareness
        - Minimum and maximum stoploss bounds
        
        Args:
            pair: Trading pair
            trade: Trade object with entry information
            current_time: Current datetime
            current_rate: Current price
            current_profit: Current profit percentage
            **kwargs: Additional keyword arguments
            
        Returns:
            float: Stoploss value as negative percentage relative to entry price
        """
        try:
            # Get the analyzed dataframe for this pair
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) == 0:
                # Fallback to fixed stoploss if no data available
                return -self.stoploss_max_percent.value
            
            # Get the most recent candle
            last_candle = dataframe.iloc[-1].squeeze()
            
            # Ensure ATR is present and valid
            if 'atr' not in last_candle or pd.isna(last_candle['atr']) or last_candle['atr'] <= 0:
                # ATR not available, use fallback based on profit status
                if current_profit > 0:
                    return -self.stoploss_min_percent.value  # Tight stoploss when profitable
                else:
                    return -self.stoploss_max_percent.value  # Wider stoploss when losing
            
            atr_value = last_candle['atr']
            atr_multiplier = self.stoploss_atr_multiplier.value
            
            # Get volatility regime for adjustment
            volatility_regime = last_candle.get('volatility_regime', 'medium')
            risk_multiplier = last_candle.get('risk_multiplier', 1.0)
            
            # Adjust ATR multiplier based on volatility regime
            if volatility_regime == 'high':
                # In high volatility, use wider stops to avoid noise
                adjusted_multiplier = atr_multiplier * 1.5
            elif volatility_regime == 'low':
                # In low volatility, use tighter stops
                adjusted_multiplier = atr_multiplier * 0.7
            else:
                # Medium volatility, use standard multiplier
                adjusted_multiplier = atr_multiplier
            
            # Calculate base stoploss distance based on ATR
            stoploss_distance = atr_value * adjusted_multiplier
            
            # Convert to percentage relative to entry price
            base_stoploss_percentage = -(stoploss_distance / trade.open_rate)
            
            # Apply profit protection logic
            if current_profit > self.stoploss_profit_protection.value:
                # When in profit above threshold, tighten the stoploss
                profit_protection_factor = 0.5  # Reduce stoploss distance by 50%
                base_stoploss_percentage = base_stoploss_percentage * profit_protection_factor
                
                # But don't make it tighter than break-even
                breakeven_stoploss = -0.001  # Small negative to account for fees
                base_stoploss_percentage = min(base_stoploss_percentage, breakeven_stoploss)
            
            # Support/Resistance level awareness
            if 'MC_Optimal_Support' in last_candle and not pd.isna(last_candle['MC_Optimal_Support']):
                support_level = last_candle['MC_Optimal_Support']
                # For long trades, don't set stoploss above the support level
                if trade.is_open and not trade.is_short:
                    support_based_stoploss = -(abs(current_rate - support_level) / trade.open_rate)
                    # Use the more conservative (wider) of the two stoplosses
                    base_stoploss_percentage = min(base_stoploss_percentage, support_based_stoploss)
            
            # Apply minimum and maximum bounds
            final_stoploss = max(base_stoploss_percentage, -self.stoploss_max_percent.value)
            final_stoploss = min(final_stoploss, -self.stoploss_min_percent.value)
            
            # Log stoploss calculation occasionally for debugging
            import random
            if random.random() < 1:  # Log 100% of the time to reduce spam
                print(f"Enhanced ATR Stoploss for {pair}:")
                print(f"  ATR={atr_value:.6f}, Base Multiplier={atr_multiplier:.1f}, Adjusted={adjusted_multiplier:.1f}")
                print(f"  Volatility Regime={volatility_regime}, Risk Multiplier={risk_multiplier:.2f}")
                print(f"  Current Profit={current_profit:.4f} ({current_profit*100:.2f}%)")
                print(f"  Base Stoploss={base_stoploss_percentage:.4f}, Final={final_stoploss:.4f}")
                print(f"  Bounds: Min={-self.stoploss_min_percent.value:.4f}, Max={-self.stoploss_max_percent.value:.4f}")
            
            return final_stoploss
                
        except Exception as e:
            # Error in custom stoploss calculation, use conservative fallback
            print(f"Error in enhanced custom_stoploss for {pair}: {e}")
            # Use tight stoploss if in profit, wide if losing
            if current_profit > 0:
                return -self.stoploss_min_percent.value
            else:
                return -self.stoploss_max_percent.value

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