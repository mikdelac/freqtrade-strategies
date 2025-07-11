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
from risk_metrics.monte_carlo import MonteCarloSimulator, MonteCarloManager
from swing_point_detector import SwingPointDetector
from trend_metrics.trendline import gentrends, segtrends, rank_trendlines, generate_bounce_conditions, Trendline
from technical.util import resample_to_interval, resampled_merge

class SignalGenerator:
    """
    Signal generation logic for bounce trading.
    Handles entry and exit signal generation with proper separation of concerns.
    """
    
    def __init__(self, strategy_instance):
        """
        Initialize SignalGenerator with reference to strategy instance.
        
        Args:
            strategy_instance: Reference to the main RiskMetrics strategy
        """
        self.strategy = strategy_instance
        
    
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
                
        # Apply trendline validity filter for long entries
        long_conditions_filtered = long_bounce_conditions & valid_trendlines
        
        # Generate short entry conditions (bounce off resistance) - using direct import from trendline.py
        short_bounce_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy.trendline_proximity_threshold.value,
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )

        # Apply trendline validity filter for short entries
        short_conditions_filtered = short_bounce_conditions & valid_trendlines
        
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
    RiskMetrics strategy focused on trendline analysis and Monte Carlo optimization.
    
    Features:
    - Fixed lookback periods for Monte Carlo optimization
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
    """
    INTERFACE_VERSION = 3

    # Timeframe settings
    timeframe = "1m"
    MINUTES_IN_DAY = 24 * 60
    MINUTES_PER_CANDLE = 1
    CANDLES_PER_DAY = MINUTES_IN_DAY // MINUTES_PER_CANDLE  # 288 5-min candles per day
    
    # Monte Carlo period optimization settings
    MC_ITERATIONS = 200
    MIN_LOOKBACK_PERIOD = 50
    # MAX_LOOKBACK_PERIOD will be set dynamically based on available data
    
    # Timeframe thresholds for resampling
    # Minimum number of candles needed for each timeframe
    TIMEFRAME_THRESHOLDS = {
        '1d': CANDLES_PER_DAY,           # Need at least 1 day of data
        '3d': CANDLES_PER_DAY * 3,       # Need at least 3 days of data
        '1w': CANDLES_PER_DAY * 5,       # Need at least 1 week of data
        '1M': CANDLES_PER_DAY * 22,      # Need at least 1 month of data
    }
    
    # Supported higher timeframes in order of preference (highest first)
    HIGHER_TIMEFRAMES = ['1M', '1w', '3d', '1d']
    
    # Trading parameters
    can_short: bool = False
    
    # Trendline parameters
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.007, space="buy", optimize=True)
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

    # Monte Carlo optimization parameters
    enable_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)
    
    # === Periodic Monte Carlo Parameters ===
    mc_recalc_interval_minutes = IntParameter(60, 480, default=120, space="buy", optimize=False)
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


    @property
    def plot_config(self):
        # Basic configuration with default plots
        plot_config = {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 12, "fillcolor": "red"},
                "all_lows": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 12, "fillcolor": "green"},
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
                    "MC_Optimal_Resistance_Period": {"color": "red", "type": "line", "width": 1.5},
                    "MC_Optimal_Support_Period": {"color": "green", "type": "line", "width": 1.5}
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
        
        # Dynamically add higher timeframe plots based on available timeframes
        if hasattr(self, 'highest_timeframe') and self.highest_timeframe != self.timeframe:
            suffix = f"_{self.highest_timeframe}"
            
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
        
        self.swing_detector = SwingPointDetector()
        
        # Initialize highest timeframe as None - will be determined dynamically
        self.highest_timeframe = None
        self.available_timeframes = []
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(self)
        
        # Initialize trendline storage for all iterations
        self.stored_trendlines = []  # List to store all Trendline objects from each Monte Carlo iteration
        
        # Initialize heartbeat tracking for live trading mode
        self.last_recalculation_time = None  # Track when we last recalculated
        self.strategy_start_time = pd.Timestamp.now()  # Track when strategy started
        self.monte_carlo_executed = False  # Track if Monte Carlo has been executed at least once
        
        # Initialize Monte Carlo Manager with required parameters
        self.monte_carlo_manager = MonteCarloManager(
            mc_iterations=self.MC_ITERATIONS,
            min_lookback_period=self.MIN_LOOKBACK_PERIOD,
            recalc_interval_minutes=self.mc_recalc_interval_minutes.value,
            trendline_proximity_threshold=self.trendline_proximity_threshold.value
        )
        
        print(f"RiskMetrics strategy initialized with Monte Carlo architecture:")
        print(f"  Lookback periods: Fixed periods for Monte Carlo optimization")
        print(f"  Monte Carlo iterations: {self.MC_ITERATIONS}")
        print(f"  Signal generator: Initialized for modular signal generation")
        print(f"  Monte Carlo Manager: Initialized with {self.mc_recalc_interval_minutes.value}min recalc interval")
        print(f"  Rolling Monte Carlo optimization: {'ENABLED' if self.enable_rolling_mc_optimization.value else 'DISABLED'}")
        print(f"  Swing Point Detector: Initialized for swing point detection")
        print(f"  Heartbeat tracking: Initialized at {self.strategy_start_time}")
        if self.enable_rolling_mc_optimization.value:
            print(f"    - Eliminates lookahead bias for realistic backtesting")
            print(f"    - Uses {self.mc_lookback_window_candles.value} candle lookback window")
            print(f"    - Recalculates every {self.mc_recalc_interval_minutes.value} minutes")
        else:
            print(f"    - Using original method (faster but with potential lookahead bias)")

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
        'daily': 252,
        'weekly': 52,
        'monthly': 12
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame.
        
        Dual-mode operation:
        1. Initial launch (backtest): Rolling Monte Carlo on all historical data
        2. Live trading (heartbeat): Non-rolling Monte Carlo only when recalc interval is reached
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Add date column if not present (required for trendline timestamps)
        if 'date' not in dataframe.columns:
            if hasattr(dataframe.index, 'to_pydatetime'):
                dataframe['date'] = dataframe.index
            else:
                # Fallback: create date column based on row count (assuming 1min candles)
                base_time = pd.Timestamp.now() - pd.Timedelta(minutes=len(dataframe))
                dataframe['date'] = pd.date_range(start=base_time, periods=len(dataframe), freq='1min')
        
        # Determine if this is initial launch (backtest) or live trading (heartbeat)
        current_time = pd.Timestamp.now()
        latest_candle_time = dataframe['date'].iloc[-1]
        
        # Ensure both timestamps have consistent timezone handling
        if latest_candle_time.tz is not None and current_time.tz is None:
            # latest_candle_time is timezone-aware, current_time is naive
            current_time = current_time.tz_localize('UTC')
        elif latest_candle_time.tz is None and current_time.tz is not None:
            # latest_candle_time is naive, current_time is timezone-aware
            latest_candle_time = latest_candle_time.tz_localize('UTC')
        elif latest_candle_time.tz is not None and current_time.tz is not None:
            # Both are timezone-aware, ensure they're in the same timezone
            if latest_candle_time.tz != current_time.tz:
                latest_candle_time = latest_candle_time.tz_convert('UTC')
                current_time = current_time.tz_convert('UTC')
        
        # Check if this is likely a backtest (historical data) or live trading (recent data)
        time_diff_minutes = (current_time - latest_candle_time).total_seconds() / 60
        
        # Multiple criteria for backtest detection:
        # 1. Data is more than 5 minutes old (historical data)
        # 2. Large amount of data (typical backtest has lots of historical candles)
        # 3. No stored trendlines yet (first run)
        # 4. Monte Carlo has never been executed (safety check)
        is_backtest_mode = (
            time_diff_minutes > 5 or  # Data is older than 5 minutes
            len(dataframe) > 10000 or  # Large dataset typical of backtests
            len(self.stored_trendlines) == 0 or  # No trendlines stored yet (first run)
            not self.monte_carlo_executed  # Monte Carlo has never been executed
        )
        
        print(f"=== MODE DETECTION ===")
        print(f"Current time: {current_time}")
        print(f"Latest candle: {latest_candle_time}")
        print(f"Time difference: {time_diff_minutes:.1f} minutes")
        print(f"Dataframe length: {len(dataframe)} candles")
        print(f"Stored trendlines: {len(self.stored_trendlines)}")
        print(f"Monte Carlo executed: {self.monte_carlo_executed}")
        print(f"Backtest criteria:")
        print(f"  - Data age > 5 min: {time_diff_minutes > 5}")
        print(f"  - Large dataset (>10k): {len(dataframe) > 10000}")
        print(f"  - No stored trendlines: {len(self.stored_trendlines) == 0}")
        print(f"  - Monte Carlo never executed: {not self.monte_carlo_executed}")
        print(f"Mode: {'BACKTEST (Rolling Monte Carlo)' if is_backtest_mode else 'LIVE TRADING (Heartbeat-based)'}")
        
        # Clear stored trendlines from previous runs only in backtest mode
        if is_backtest_mode:
            self.stored_trendlines.clear()
            print(f"Cleared previous trendline storage for backtest analysis of {metadata.get('pair', 'UNKNOWN')}")
        else:
            print(f"Keeping existing trendlines for live trading of {metadata.get('pair', 'UNKNOWN')} ({len(self.stored_trendlines)} stored)")
            
        # Determine the highest timeframe we can use based on available data
        self.highest_timeframe = self.determine_highest_timeframe(dataframe)
        
        # Add an indicator showing which highest timeframe was selected
        dataframe.loc[:, 'highest_timeframe_indicator'] = 1.0
        dataframe.loc[:, 'highest_timeframe'] = self.highest_timeframe
        print(f"Highest timeframe: {self.highest_timeframe}")
        
        # Resample to higher timeframes if possible
        resampled_dfs = self.resample_to_higher_timeframes(dataframe)
        print(f"Resampled dataframes: {resampled_dfs}")

        # Initialize all required dataframe columns
        self._initialize_dataframe_columns(dataframe)

        if is_backtest_mode:
            # === BACKTEST MODE: Rolling Monte Carlo ===
            print("=== EXECUTING BACKTEST MODE ===")
            
            # In backtest mode, always run Monte Carlo optimization since we cleared stored trendlines
            # Execute rolling Monte Carlo optimization for backtest
            if self.enable_rolling_mc_optimization.value:
                self._execute_rolling_monte_carlo_optimization(dataframe, metadata)
                self._update_last_recalculation_time(latest_candle_time)
            else:
                self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)
            
            # Mark that Monte Carlo has been executed
            self.monte_carlo_executed = True
        else:
            # === LIVE TRADING MODE: Heartbeat-based Monte Carlo ===
            print("=== EXECUTING LIVE TRADING MODE ===")
            
            # Check if we need to recalculate based on time interval
            should_recalculate = self._should_recalculate_for_heartbeat(latest_candle_time)
            
            if should_recalculate:
                print(f"Recalculation interval reached - executing non-rolling Monte Carlo")
                self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)
                # Update last recalculation time
                self._update_last_recalculation_time(latest_candle_time)
                # Mark that Monte Carlo has been executed
                self.monte_carlo_executed = True
            else:
                print(f"Using existing trendlines - no recalculation needed")
                # Try to populate from existing trendlines
                if not self._populate_from_existing_trendlines(dataframe, metadata):
                    print(f"No existing trendlines found - forcing recalculation")
                    self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)
                    self._update_last_recalculation_time(latest_candle_time)
                    # Mark that Monte Carlo has been executed
                    self.monte_carlo_executed = True

        # Mark high and low points for visualization
        try:
            # Find and map swing points to both dataframes using SwingPointDetector
            self.swing_detector.find_and_map_swing_points(dataframe, dataframe)
            
        except Exception as e:
            print(f"Error finding swing points: {e}")

        # === Output All Stored Trendlines ===
        try:
            print("\n=== STORED TRENDLINES SUMMARY ===")
            pair = metadata.get('pair', 'UNKNOWN')
            
            # Count trendlines from stored_trendlines list
            total_trendlines = len(self.stored_trendlines)
            resistance_trendlines = sum(1 for tl in self.stored_trendlines if tl.trendline_type == 'resistance')
            support_trendlines = sum(1 for tl in self.stored_trendlines if tl.trendline_type == 'support')
            
            print(f"Pair: {pair}")
            print(f"Monte Carlo Results Available: Yes")
            
            # Display trendline count summary
            print(f"\n--- TRENDLINE COUNT SUMMARY ---")
            print(f"  Total Saved Trendlines: {total_trendlines}")
            print(f"  Resistance Trendlines: {resistance_trendlines}")
            print(f"  Support Trendlines: {support_trendlines}")
            
            # Display all stored trendlines
            if total_trendlines > 0:
                print(f"\n--- ALL STORED TRENDLINES ---")
                for i, trendline in enumerate(self.stored_trendlines, 1):
                    print(f"  {i}. {trendline.trendline_type.upper()} TRENDLINE")
                    print(f"     Start Time: {trendline.start_time}")
                    print(f"     End Time: {trendline.end_time}")
                    print(f"     Duration: {trendline.duration_hours:.2f} hours")
                    print(f"     Age: {trendline.age_hours:.2f} hours")
                    print(f"     Slope: {trendline.slope:.8f}")
                    print(f"     R-squared: {trendline.r_squared:.4f}")
                    print(f"     Start Price: {trendline.start_price:.6f}")
                    print(f"     Bounce Count: {trendline.bounce_count}")
                    
                    # Display bounce timestamps if available
                    if hasattr(trendline, 'bounce_timestamps') and trendline.bounce_timestamps:
                        print(f"     Bounce Timestamps ({len(trendline.bounce_timestamps)}):")
                        for j, timestamp in enumerate(trendline.bounce_timestamps, 1):
                            print(f"       {j}. {timestamp}")
                    else:
                        print(f"     Bounce Timestamps: None")
                    
                    print("")
            else:
                print(f"  No trendlines stored during this execution")
            
            # Show current dataframe trendline values for the latest candle
            if len(dataframe) > 0:
                latest_candle = dataframe.iloc[-1]
                print(f"\n--- CURRENT DATAFRAME VALUES (Latest Candle) ---")
                print(f"  MC_Optimal_Resistance: {latest_candle.get('MC_Optimal_Resistance', np.nan):.6f}")
                print(f"  MC_Optimal_Support: {latest_candle.get('MC_Optimal_Support', np.nan):.6f}")
                print(f"  MC_Resistance_Score: {latest_candle.get('MC_Resistance_Score', 0.0):.6f}")
                print(f"  MC_Support_Score: {latest_candle.get('MC_Support_Score', 0.0):.6f}")
                print(f"  Close Price: {latest_candle.get('close', np.nan):.6f}")
                
                # Calculate distances to trendlines
                resistance_price = latest_candle.get('MC_Optimal_Resistance', np.nan)
                support_price = latest_candle.get('MC_Optimal_Support', np.nan)
                close_price = latest_candle.get('close', np.nan)
                
                if not np.isnan(resistance_price) and not np.isnan(close_price):
                    resistance_distance = ((resistance_price - close_price) / close_price) * 100
                    print(f"  Distance to Resistance: {resistance_distance:+.3f}%")
                
                if not np.isnan(support_price) and not np.isnan(close_price):
                    support_distance = ((support_price - close_price) / close_price) * 100
                    print(f"  Distance to Support: {support_distance:+.3f}%")
            
            print(f"=== END TRENDLINES SUMMARY ===\n")
            
        except Exception as e:
            print(f"Error outputting stored trendlines: {e}")
            
        return dataframe

    def _initialize_dataframe_columns(self, dataframe: DataFrame) -> None:
        """
        Initialize all required columns in the dataframe with default values.
        
        Args:
            dataframe: DataFrame to initialize columns in
        """
        # Initialize marker columns for support and resistance points using proper pandas assignment
        dataframe.loc[:, 'all_highs'] = np.nan
        dataframe.loc[:, 'all_lows'] = np.nan
                
        # Initialize Monte Carlo optimal lines
        dataframe.loc[:, 'MC_Optimal_Resistance'] = np.nan
        dataframe.loc[:, 'MC_Optimal_Support'] = np.nan
        dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
        dataframe.loc[:, 'MC_Support_Score'] = 0.0
        dataframe.loc[:, 'MC_Optimal_Period'] = 0.0
                
    def _populate_from_existing_trendlines(self, dataframe: DataFrame, metadata: dict) -> bool:
        """
        Draw every stored trendline during their exact start/end time periods.
        
        Args:
            dataframe: DataFrame to populate with existing trendline data
            metadata: Strategy metadata containing pair information
            
        Returns:
            bool: True if any trendlines were drawn, False otherwise
        """
        if not self.stored_trendlines:
            return False
        
        pair = metadata.get('pair', 'UNKNOWN')
        print(f"Drawing {len(self.stored_trendlines)} stored trendlines for {pair}")
        
        # Initialize all columns with NaN
        dataframe.loc[:, 'MC_Optimal_Resistance'] = np.nan
        dataframe.loc[:, 'MC_Optimal_Support'] = np.nan
        dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
        dataframe.loc[:, 'MC_Support_Score'] = 0.0
        
        trendlines_drawn = False
        
        # Draw each stored trendline during its exact time period
        for trendline in self.stored_trendlines:
            for j, timestamp in enumerate(dataframe['date']):
                pandas_index = dataframe.index[j]
                
                # Only draw trendline if timestamp is within its exact active period
                if trendline.is_active_at_time(timestamp):
                    price = trendline.get_price_at_time(timestamp)
                    
                    if trendline.trendline_type == 'resistance':
                        dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = price
                        dataframe.loc[pandas_index, 'MC_Resistance_Score'] = float(trendline.bounce_count)
                    elif trendline.trendline_type == 'support':
                        dataframe.loc[pandas_index, 'MC_Optimal_Support'] = price
                        dataframe.loc[pandas_index, 'MC_Support_Score'] = float(trendline.bounce_count)
                    
                    trendlines_drawn = True
        
        if trendlines_drawn:
            print(f"Successfully drew stored trendlines for {pair}")
        else:
            print(f"No trendlines were active during dataframe period for {pair}")
        
        return trendlines_drawn

    def _apply_trendline_to_dataframe(self, dataframe: DataFrame, trendline: Optional[Trendline], trendline_type: str) -> None:
        """
        Apply a single trendline to the dataframe by calculating its price at each timestamp.
        
        Args:
            dataframe: DataFrame to populate
            trendline: Trendline object to apply (can be None)
            trendline_type: Either 'resistance' or 'support'
        """
        if trendline is None:
            # Set columns to NaN if no trendline available
            if trendline_type == 'resistance':
                dataframe.loc[:, 'MC_Optimal_Resistance'] = np.nan
                dataframe.loc[:, 'MC_Resistance_Score'] = 0.0
                dataframe.loc[:, 'MC_Optimal_Resistance_Period'] = 0.0
            else:  # support
                dataframe.loc[:, 'MC_Optimal_Support'] = np.nan
                dataframe.loc[:, 'MC_Support_Score'] = 0.0
                dataframe.loc[:, 'MC_Optimal_Support_Period'] = 0.0
            return
        
        # Calculate trendline price at each dataframe timestamp
        trendline_prices = []
        for timestamp in dataframe['date']:
            if trendline.is_active_at_time(timestamp):
                price = trendline.get_price_at_time(timestamp)
                trendline_prices.append(price)
            else:
                # Extend trendline beyond its original range using the slope
                price = trendline.get_price_at_time(timestamp)
                trendline_prices.append(price)
        
        # Apply to appropriate columns
        if trendline_type == 'resistance':
            dataframe.loc[:, 'MC_Optimal_Resistance'] = trendline_prices
            dataframe.loc[:, 'MC_Resistance_Score'] = float(trendline.bounce_count)
            # Calculate period from trendline duration (convert to number of candles)
            duration_minutes = trendline.duration_hours * 60
            timeframe_minutes = timeframe_to_minutes(self.timeframe)
            estimated_period = int(duration_minutes / timeframe_minutes) if timeframe_minutes > 0 else 0
            dataframe.loc[:, 'MC_Optimal_Resistance_Period'] = float(estimated_period)
        else:  # support
            dataframe.loc[:, 'MC_Optimal_Support'] = trendline_prices
            dataframe.loc[:, 'MC_Support_Score'] = float(trendline.bounce_count)
            # Calculate period from trendline duration (convert to number of candles)
            duration_minutes = trendline.duration_hours * 60
            timeframe_minutes = timeframe_to_minutes(self.timeframe)
            estimated_period = int(duration_minutes / timeframe_minutes) if timeframe_minutes > 0 else 0
            dataframe.loc[:, 'MC_Optimal_Support_Period'] = float(estimated_period)

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


    def _execute_rolling_monte_carlo_recalculation(self, dataframe: DataFrame, i: int, metadata: dict) -> Optional[Dict]:
        """
        Execute Monte Carlo recalculation for a specific candle index.
        
        Args:
            dataframe: The full dataframe
            i: Current candle index
            metadata: Strategy metadata
            
        Returns:
            Dictionary containing Monte Carlo results or None if failed
        """
        # Define the lookback window for the current candle (point-in-time data only)
        lookback_start = max(0, i - self.mc_lookback_window_candles.value)
        current_dataframe_slice = dataframe.iloc[lookback_start:i].copy()

        # Find and map swing points for the current slice
        high_swing_points = self.swing_detector.find_swing_points(
            prices=dataframe['high'].values,
            price_type='high'
        )
        low_swing_points = self.swing_detector.find_swing_points(
            prices=dataframe['low'].values,
            price_type='low'
        )

        current_dataframe_slice.loc[:, 'all_highs'] = np.nan
        current_dataframe_slice.loc[:, 'all_lows'] = np.nan

        for idx, price in high_swing_points:
            if lookback_start <= idx < i:
                slice_idx = idx - lookback_start
                current_dataframe_slice.iloc[slice_idx, current_dataframe_slice.columns.get_loc('all_highs')] = price
        
        for idx, price in low_swing_points:
            if lookback_start <= idx < i:
                slice_idx = idx - lookback_start
                current_dataframe_slice.iloc[slice_idx, current_dataframe_slice.columns.get_loc('all_lows')] = price

        print(f"  Using data slice: {lookback_start} to {i} ({len(current_dataframe_slice)} candles)")

        # Execute MC optimization on the slice of data available at this point in time
        # Create a temporary manager to ensure no state from future data is used
        temp_mc_manager = MonteCarloManager(
            mc_iterations=self.MC_ITERATIONS,
            min_lookback_period=self.MIN_LOOKBACK_PERIOD,
            recalc_interval_minutes=0,  # Force recalc for temporary manager
            trendline_proximity_threshold=self.trendline_proximity_threshold.value
        )
        
        mc_results = temp_mc_manager.execute_monte_carlo_optimization(
            current_dataframe_slice, metadata['pair'], self.enable_mc_optimization.value,
            actual_recalc_interval_minutes=self.mc_recalc_interval_minutes.value
        )
        
        return mc_results

    def _update_slopes_and_values_from_mc_results(self, mc_results: Dict) -> Tuple[float, float, float, float]:
        """
        Update resistance and support slopes and values from Monte Carlo results.
        
        Args:
            mc_results: Dictionary containing Monte Carlo optimization results
            
        Returns:
            Tuple of (current_resistance_slope, current_support_slope, last_resistance_value, last_support_value)
        """
        # Update slopes from the new MC results
        current_resistance_slope = mc_results.get('best_resistance_slope', 0.0)
        current_support_slope = mc_results.get('best_support_slope', 0.0)
        
        # Get the last valid value from the calculated lines as the starting point
        res_line = mc_results.get('resistance_line', np.array([]))
        valid_res = res_line[~np.isnan(res_line)]
        last_resistance_value = valid_res[-1] if len(valid_res) > 0 else np.nan

        sup_line = mc_results.get('support_line', np.array([]))
        valid_sup = sup_line[~np.isnan(sup_line)]
        last_support_value = valid_sup[-1] if len(valid_sup) > 0 else np.nan
        
        return current_resistance_slope, current_support_slope, last_resistance_value, last_support_value

    def _project_trendlines_forward(self, dataframe: DataFrame, i: int, 
                                   last_resistance_value: float, last_support_value: float,
                                   current_resistance_slope: float, current_support_slope: float,
                                   last_mc_results: Optional[Dict]) -> Tuple[float, float]:
        """
        Project trendlines forward using slopes and apply results to dataframe.
        
        Args:
            dataframe: The dataframe to update
            i: Current candle index
            last_resistance_value: Current resistance value
            last_support_value: Current support value
            current_resistance_slope: Resistance slope
            current_support_slope: Support slope
            last_mc_results: Latest Monte Carlo results
            
        Returns:
            Tuple of updated (last_resistance_value, last_support_value)
        """
        pandas_index = dataframe.index[i]
        
        # Project resistance line forward
        if not np.isnan(last_resistance_value):
            last_resistance_value += current_resistance_slope
            dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = last_resistance_value
        else:
            dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = np.nan
        
        # Project support line forward
        if not np.isnan(last_support_value):
            last_support_value += current_support_slope
            dataframe.loc[pandas_index, 'MC_Optimal_Support'] = last_support_value
        else:
            dataframe.loc[pandas_index, 'MC_Optimal_Support'] = np.nan
        
        # Apply scores to this specific row
        dataframe.loc[pandas_index, 'MC_Resistance_Score'] = last_mc_results.get('resistance_score', 0.0) if last_mc_results else 0.0
        dataframe.loc[pandas_index, 'MC_Support_Score'] = last_mc_results.get('support_score', 0.0) if last_mc_results else 0.0
        dataframe.loc[pandas_index, 'MC_Optimal_Resistance_Period'] = last_mc_results.get('optimal_resistance_period', 0.0) if last_mc_results else 0.0
        dataframe.loc[pandas_index, 'MC_Optimal_Support_Period'] = last_mc_results.get('optimal_support_period', 0.0) if last_mc_results else 0.0
        
        return last_resistance_value, last_support_value

    def _initialize_early_candles(self, dataframe: DataFrame, min_required_candles: int) -> None:
        """
        Initialize the early candles that couldn't be processed with rolling optimization.
        
        Args:
            dataframe: The dataframe to initialize
            min_required_candles: Number of candles that need initialization
        """
        for i in range(0, min_required_candles):
            pandas_index = dataframe.index[i]
            dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = np.nan
            dataframe.loc[pandas_index, 'MC_Optimal_Support'] = np.nan
            dataframe.loc[pandas_index, 'MC_Resistance_Score'] = 0.0
            dataframe.loc[pandas_index, 'MC_Support_Score'] = 0.0
            dataframe.loc[pandas_index, 'MC_Optimal_Resistance_Period'] = 0.0
            dataframe.loc[pandas_index, 'MC_Optimal_Support_Period'] = 0.0
            dataframe.loc[pandas_index, 'score_convergence_ratio'] = 0.0
            dataframe.loc[pandas_index, 'convergence_multiplier'] = 1.0
            dataframe.loc[pandas_index, 'trading_mode_indicator'] = 0.0
            dataframe.loc[pandas_index, 'trading_mode'] = "INSUFFICIENT_DATA"

    def _execute_rolling_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute rolling Monte Carlo optimization to eliminate lookahead bias.
        Skip simulations where valid trendlines already exist for the time period.
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        # Calculate intervals and thresholds
        recalc_interval_candles = self.mc_recalc_interval_minutes.value // timeframe_to_minutes(self.timeframe)
        min_required_candles = max(self.MIN_LOOKBACK_PERIOD, 100)
        
        print("=== Using Rolling Monte Carlo Optimization (eliminates lookahead bias) ===")
        print(f"=== Starting Rolling Monte Carlo Analysis for {metadata['pair']} ===")
        print(f"Recalc interval: {recalc_interval_candles} candles ({self.mc_recalc_interval_minutes.value} minutes)")
        print(f"Lookback window: {self.mc_lookback_window_candles.value} candles")
        print(f"Processing ALL {len(dataframe) - min_required_candles} candles starting from minimum required data ({min_required_candles})")

        # State variables for projecting trendlines with slopes
        last_resistance_value = np.nan
        last_support_value = np.nan
        current_resistance_slope = 0.0
        current_support_slope = 0.0
        last_mc_results = None

        # Main rolling optimization loop
        for i in range(min_required_candles, len(dataframe)):
            current_time = dataframe['date'].iloc[i]  # Current timestamp being processed
            
            # Determine if it is time to recalculate
            should_recalculate = (i == min_required_candles) or ((i - min_required_candles) % recalc_interval_candles == 0)

            if should_recalculate:
                # Check if we already have valid trendlines for this time period
                has_valid_resistance = False
                has_valid_support = False
                
                for trendline in self.stored_trendlines:
                    if trendline.is_active_at_time(current_time):
                        if trendline.trendline_type == 'resistance':
                            has_valid_resistance = True
                        elif trendline.trendline_type == 'support':
                            has_valid_support = True
                        
                        # If we have both types, we can skip this simulation
                        if has_valid_resistance and has_valid_support:
                            break
                
                # If we don't have active trendlines, look for the closest future trendlines
                if not (has_valid_resistance and has_valid_support):
                    print(f"Looking for closest future trendlines for time {current_time}")
                    
                    # Find closest future resistance trendline (end_time > current_time)
                    closest_future_resistance = None
                    min_resistance_time_diff = None
                    
                    # Find closest future support trendline (end_time > current_time)
                    closest_future_support = None
                    min_support_time_diff = None
                    
                    for trendline in self.stored_trendlines:
                        # Only consider trendlines that end after current time
                        if trendline.end_time > current_time:
                            time_diff = (trendline.end_time - current_time).total_seconds()
                            
                            if trendline.trendline_type == 'resistance' and not has_valid_resistance:
                                if min_resistance_time_diff is None or time_diff < min_resistance_time_diff:
                                    closest_future_resistance = trendline
                                    min_resistance_time_diff = time_diff
                                    print(f"    Found future resistance candidate: end_time={trendline.end_time}, time_diff={time_diff/60:.1f}min")
                            elif trendline.trendline_type == 'support' and not has_valid_support:
                                if min_support_time_diff is None or time_diff < min_support_time_diff:
                                    closest_future_support = trendline
                                    min_support_time_diff = time_diff
                                    print(f"    Found future support candidate: end_time={trendline.end_time}, time_diff={time_diff/60:.1f}min")
                    
                    # Assign closest future trendlines if found
                    if closest_future_resistance and not has_valid_resistance:
                        last_resistance_value = closest_future_resistance.get_price_at_time(current_time)
                        current_resistance_slope = closest_future_resistance.slope
                        has_valid_resistance = True
                        print(f"  ✓ Assigned closest future resistance: end_time={closest_future_resistance.end_time}, price={last_resistance_value:.6f}, slope={current_resistance_slope:.6f}")
                    
                    if closest_future_support and not has_valid_support:
                        last_support_value = closest_future_support.get_price_at_time(current_time)
                        current_support_slope = closest_future_support.slope
                        has_valid_support = True
                        print(f"  ✓ Assigned closest future support: end_time={closest_future_support.end_time}, price={last_support_value:.6f}, slope={current_support_slope:.6f}")
                    
                    # Debug: Show what we found vs what we were looking for
                    if not has_valid_resistance:
                        print(f"  ✗ No future resistance found for time {current_time}")
                    if not has_valid_support:
                        print(f"  ✗ No future support found for time {current_time}")
                
                # Skip simulation if we now have both resistance and support trendlines
                if has_valid_resistance and has_valid_support:
                    print(f"Skipping MC simulation at candle {i}/{len(dataframe)} - using stored/assigned trendlines for time {current_time}")
                    
                    # Use existing trendlines to update state variables (only if not already set above)
                    for trendline in self.stored_trendlines:
                        if trendline.is_active_at_time(current_time):
                            if trendline.trendline_type == 'resistance' and np.isnan(last_resistance_value):
                                last_resistance_value = trendline.get_price_at_time(current_time)
                                current_resistance_slope = trendline.slope
                                print(f"  Using existing active resistance: price={last_resistance_value:.6f}, slope={current_resistance_slope:.6f}")
                            elif trendline.trendline_type == 'support' and np.isnan(last_support_value):
                                last_support_value = trendline.get_price_at_time(current_time)
                                current_support_slope = trendline.slope
                                print(f"  Using existing active support: price={last_support_value:.6f}, slope={current_support_slope:.6f}")
                else:
                    # Execute Monte Carlo recalculation only if we still don't have valid trendlines
                    print(f"Recalculating MC results at candle {i}/{len(dataframe)} ({(i/len(dataframe)*100):.1f}%) - missing trendlines for time {current_time}")
                    
                    # Debug: List all stored trendlines before recalculation
                    print(f"=== DEBUG: All Stored Trendlines at {current_time} ===")
                    print(f"Total stored trendlines: {len(self.stored_trendlines)}")
                    for idx, tl in enumerate(self.stored_trendlines, 1):
                        is_active = tl.is_active_at_time(current_time)
                        print(f"  {idx}. {tl.trendline_type.upper()} - Start: {tl.start_time}, End: {tl.end_time}, Active: {is_active}")
                    print(f"=== END DEBUG ===")
                    
                    mc_results = self._execute_rolling_monte_carlo_recalculation(dataframe, i, metadata)
                    
                    if mc_results:
                        last_mc_results = mc_results
                        
                        # Store trendline objects from this iteration
                        resistance_trendline = mc_results.get('best_resistance_trendline')
                        support_trendline = mc_results.get('best_support_trendline')
                        
                        if resistance_trendline:
                            self.stored_trendlines.append(resistance_trendline)
                            print(f"  Stored resistance trendline: {resistance_trendline.trendline_type} at candle {i}")
                        
                        if support_trendline:
                            self.stored_trendlines.append(support_trendline)
                            print(f"  Stored support trendline: {support_trendline.trendline_type} at candle {i}")
                        
                        # Update slopes and values
                        (current_resistance_slope, current_support_slope, 
                         last_resistance_value, last_support_value) = self._update_slopes_and_values_from_mc_results(mc_results)
                        
                        print(f"  New MC results: R_score={mc_results.get('resistance_score', 0):.4f}, S_score={mc_results.get('support_score', 0):.4f}")
                        print(f"  Slopes: R_slope={current_resistance_slope:.6f}, S_slope={current_support_slope:.6f}")

            # Project trendlines forward using slopes
            last_resistance_value, last_support_value = self._project_trendlines_forward(
                dataframe, i, last_resistance_value, last_support_value,
                current_resistance_slope, current_support_slope, last_mc_results
            )

        # Final processing
        print(f"Rolling Monte Carlo optimization completed for {metadata['pair']}")
        if last_mc_results:
            print(f"Final results: R_score={last_mc_results.get('resistance_score', 0):.4f}, S_score={last_mc_results.get('support_score', 0):.4f}")
            print(f"Final slopes: R_slope={current_resistance_slope:.6f}, S_slope={current_support_slope:.6f}")
            
        # Initialize early candles
        self._initialize_early_candles(dataframe, min_required_candles)

    def _execute_non_rolling_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute non-rolling Monte Carlo optimization (original method with potential lookahead bias).
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        print("=== Using Original Monte Carlo Optimization (with potential lookahead bias) ===")
        
        # Find and map swing points
        high_swing_points = self.swing_detector.find_swing_points(
            prices=dataframe['high'].values,
            price_type='high'
        )
        low_swing_points = self.swing_detector.find_swing_points(
            prices=dataframe['low'].values,
            price_type='low'
        )

        dataframe.loc[:, 'all_highs'] = np.nan
        dataframe.loc[:, 'all_lows'] = np.nan

        for idx, price in high_swing_points:
            if idx < len(dataframe):
                dataframe.iloc[idx, dataframe.columns.get_loc('all_highs')] = price
        
        for idx, price in low_swing_points:
            if idx < len(dataframe):
                dataframe.iloc[idx, dataframe.columns.get_loc('all_lows')] = price

        # Execute Monte Carlo optimization using the manager (original method)
        mc_results = self.monte_carlo_manager.execute_monte_carlo_optimization(
            dataframe, metadata['pair'], self.enable_mc_optimization.value,
            actual_recalc_interval_minutes=self.mc_recalc_interval_minutes.value
        )
        
        # Apply the results to the dataframe
        self.monte_carlo_manager.apply_monte_carlo_results(dataframe, mc_results)
        
        if mc_results:
            print(f"Applied Monte Carlo results for {metadata['pair']}:")
            print(f"  Resistance Score: {mc_results.get('resistance_score', 0.0):.6f}")
            print(f"  Support Score: {mc_results.get('support_score', 0.0):.6f}")
            print(f"  Optimal Periods: R={mc_results.get('optimal_resistance_period', 0)}, S={mc_results.get('optimal_support_period', 0)}")
            
            # Store the best trendline objects returned by Monte Carlo optimization
            resistance_trendline = mc_results.get('best_resistance_trendline')
            support_trendline = mc_results.get('best_support_trendline')
            
            if resistance_trendline:
                self.stored_trendlines.append(resistance_trendline)
                print(f"  Stored best resistance trendline from Monte Carlo optimization")
            
            if support_trendline:
                self.stored_trendlines.append(support_trendline)
                print(f"  Stored best support trendline from Monte Carlo optimization")
        else:
            print(f"Monte Carlo results not yet available for {metadata['pair']}.")

    def _should_recalculate_for_heartbeat(self, current_candle_time: pd.Timestamp) -> bool:
        """
        Determine if we should recalculate Monte Carlo based on heartbeat interval.
        
        Uses modulo logic to check if enough time has passed since strategy start
        or last recalculation.
        
        Args:
            current_candle_time: Current candle timestamp
            
        Returns:
            bool: True if recalculation is needed
        """
        recalc_interval_minutes = self.mc_recalc_interval_minutes.value
        
        # Ensure consistent timezone handling
        last_recalc_time = self.last_recalculation_time
        current_time = current_candle_time
                
        # Calculate minutes since last recalculation
        minutes_since_last_recalc = (current_time - last_recalc_time).total_seconds() / 60
        
        # Check if we've reached the recalc interval using modulo logic
        should_recalc = minutes_since_last_recalc % recalc_interval_minutes == 0
        
        print(f"=== HEARTBEAT RECALCULATION CHECK ===")
        print(f"Current candle time: {current_time}")
        print(f"Last recalculation: {last_recalc_time}")
        print(f"Minutes since last recalc: {minutes_since_last_recalc:.1f}")
        print(f"Recalc interval: {recalc_interval_minutes} minutes")
        print(f"Should recalculate: {should_recalc}")
        
        return should_recalc
    
    def _update_last_recalculation_time(self, current_candle_time: pd.Timestamp) -> None:
        """
        Update the last recalculation time to the current candle time.
        
        Args:
            current_candle_time: Current candle timestamp
        """
        self.last_recalculation_time = current_candle_time
        print(f"Updated last recalculation time to: {self.last_recalculation_time}")

