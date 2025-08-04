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
import talib.abstract as ta

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
from risk_metrics.monte_carlo import MonteCarloSimulator
from swing_point_detector import SwingPointDetector
from trend_metrics.trendline import (
    gentrends, segtrends, rank_trendlines, generate_bounce_conditions, 
    Trendline, output_trendlines_info, monte_carlo_period_optimization,
    evaluate_lookback_period_for_trendlines, generate_lookback_periods,
    project_trendlines_forward, generate_forward_trendline
)
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
        Generate entry signals based on RSI crossing support from below.
        Filters signals using Bollinger Bands for confirmation.
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame: Updated dataframe with entry signals
        """
        # Initialize entry signals
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Check if required columns exist
        required_columns = ['rsi', 'RSI_Optimal_Support', 'RSI_Optimal_Resistance', 
                          'bb_lowerband', 'bb_middleband', 'bb_upperband']
        if not all(col in dataframe.columns for col in required_columns):
            print("Missing required columns for RSI signal generation")
            return dataframe
        
        # Create RSI crossover conditions
        rsi_support_crossover = (
            (dataframe['rsi'] > dataframe['RSI_Optimal_Support']) &  # Current RSI above support
            (dataframe['rsi'].shift(1) <= dataframe['RSI_Optimal_Support'].shift(1)) &  # Previous RSI was below support
            (~dataframe['RSI_Optimal_Support'].isna()) &  # Support line exists
            (~dataframe['RSI_Optimal_Support'].shift(1).isna()) &  # Previous support line exists
            (dataframe['close'] < dataframe['bb_middleband']) &  # Price below BB middle band
            (dataframe['close'] > dataframe['bb_lowerband'])  # Price above BB lower band
        )
        
        # Set entry signals
        dataframe.loc[rsi_support_crossover, 'enter_long'] = 1
        
        # Log entry signal summary
        long_signals = dataframe['enter_long'].sum()
        print(f"RSI-based entry signals generated: {long_signals} long")
        
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
            dataframe['MC_Optimal_Resistance'], 'short', self.strategy._get_trendline_proximity_threshold(),
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        
        # Generate long entry conditions for short exits - using direct import from trendline.py
        long_entry_conditions = generate_bounce_conditions(
            dataframe['close'], 
            dataframe['MC_Optimal_Support'], 'long', self.strategy._get_trendline_proximity_threshold(),
            dataframe.get('all_highs'), dataframe.get('all_lows')
        )
        

        short_entry_filtered = short_entry_conditions
        long_entry_filtered = long_entry_conditions
        
        return short_entry_filtered, long_entry_filtered
    
    def generate_exit_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Generate exit signals based on RSI approaching resistance.
        Uses Bollinger Bands for additional confirmation.
        
        Args:
            dataframe: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame: Updated dataframe with exit signals
        """
        # Initialize exit signals
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Check if required columns exist
        required_columns = ['rsi', 'RSI_Optimal_Resistance', 'bb_upperband']
        if not all(col in dataframe.columns for col in required_columns):
            print("Missing required columns for RSI exit signal generation")
            return dataframe
        
        # Define proximity threshold for RSI resistance (within 5 points)
        rsi_resistance_proximity = 5.0
        
        # Create RSI resistance proximity condition with BB confirmation
        near_rsi_resistance = (
            (dataframe['RSI_Optimal_Resistance'] - dataframe['rsi'] <= rsi_resistance_proximity) &
            (dataframe['rsi'] < dataframe['RSI_Optimal_Resistance']) &  # RSI below resistance
            (~dataframe['RSI_Optimal_Resistance'].isna()) &  # Resistance line exists
            (dataframe['close'] > dataframe['bb_upperband'])  # Price above BB upper band
        )
        
        # Set exit signals
        dataframe.loc[near_rsi_resistance, 'exit_long'] = 1
        
        # Log exit signal summary
        long_exits = dataframe['exit_long'].sum()
        print(f"RSI-based exit signals generated: {long_exits} long exits")
        
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
    timeframe = "5m"
    MINUTES_IN_DAY = 24 * 60
    # CANDLES_PER_DAY will be calculated dynamically in __init__ based on actual timeframe
    
    # Monte Carlo period optimization settings
    MC_ITERATIONS = 200
    MIN_LOOKBACK_PERIOD = 50
    # MAX_LOOKBACK_PERIOD will be set dynamically based on available data
            
    # Trading parameters
    can_short: bool = False
    
    # Trendline parameters
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.1, space="buy", optimize=True)
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

    # Monte Carlo optimization parameters
    enable_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)
    
    # === Periodic Monte Carlo Parameters ===
    mc_recalc_interval_minutes = IntParameter(60, 480, default=240, space="buy", optimize=False)
    mc_lookback_window_candles = IntParameter(1000, 3000, default=2000, space="buy", optimize=False)
    
    # Rolling Monte Carlo optimization (eliminates lookahead bias)
    enable_rolling_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)

    # RSI parameters
    rsi_timeperiod = IntParameter(10, 30, default=14, space="buy", optimize=True)
    
    # === RSI Trendline Parameters ===
    enable_rsi_trendlines = BooleanParameter(default=True, space="buy", optimize=False)
    rsi_trendline_proximity_threshold = DecimalParameter(0.5, 5.0, default=2.0, space="buy", optimize=True)

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

    # Timeframe hierarchy for recursive analysis
    TIMEFRAME_HIERARCHY = ['5m', '1h', '4h', '1d', '1w']
    
    # SwingPointDetector parameters for each timeframe
    SWING_DETECTOR_PARAMS = {
        '5m': {  # Default timeframe
            'distance': 20,     # Minimum distance between swing points (10 candles)
            'prominence': 0.02, # 2% prominence for 5m timeframe
            'wlen': None,
            'width': None
        },
        '1h': {
            'distance': 8,      # Minimum distance between swing points (8 candles)
            'prominence': 0.03, # 3% prominence for 1h timeframe
            'wlen': None,       # No width limit
            'width': None       # No width limit
        },
        '4h': {
            'distance': 6,      # Minimum distance between swing points (6 candles)
            'prominence': 0.04, # 4% prominence for 4h timeframe
            'wlen': None,
            'width': None
        },
        '1d': {
            'distance': 5,      # Minimum distance between swing points (5 candles)
            'prominence': 0.04, # 4% prominence for 1d timeframe
            'wlen': None,
            'width': None
        },
        '1w': {
            'distance': 4,      # Minimum distance between swing points (4 candles)
            'prominence': 0.04, # 4% prominence for 1w timeframe
            'wlen': None,
            'width': None
        }
    }
    
    # Trendline proximity thresholds for each timeframe
    TRENDLINE_PROXIMITY_THRESHOLDS = {
        '5m': 0.001,   # 0.1% proximity threshold for 5m timeframe (tight)
        '1h': 0.004,   # 0.4% proximity threshold for 1h timeframe
        '4h': 0.012,   # 1.2% proximity threshold for 4h timeframe
        '1d': 0.075,   # 7.5% proximity threshold for 1d timeframe
        '1w': 0.20,    # 20.0% proximity threshold for 1w timeframe (loose)
    }

    @property
    def plot_config(self):
        # Basic configuration with default plots
        plot_config = {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 12, "fillcolor": "red"},
                "all_lows": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 12, "fillcolor": "green"},
                "MC_Optimal_Resistance": {"color": "darkred", "width": 4.0, "dash": "dot"},
                "MC_Optimal_Support": {"color": "darkgreen", "width": 4.0, "dash": "dot"},
                "bb_upperband": {"color": "rgba(255,144,144,0.6)", "fill": None, "width": 1.0},
                "bb_middleband": {"color": "rgba(144,144,144,0.6)", "width": 1.0},
                "bb_lowerband": {"color": "rgba(144,255,144,0.6)", "fill": "tonexty", "width": 1.0},
                "bb_high_tag": {"color": "yellow", "type": "scatter", "symbol": "triangle-down", "size": 10, "fillcolor": "yellow"},
                "bb_low_tag": {"color": "blue", "type": "scatter", "symbol": "triangle-up", "size": 10, "fillcolor": "blue"}
            },
            "subplots": {
                "RSI": {
                    "rsi": {"color": "purple", "type": "line", "width": 2.0},
                    "all_highs_rsi": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 8, "fillcolor": "red"},
                    "all_lows_rsi": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 8, "fillcolor": "green"},
                    "RSI_Optimal_Resistance": {"color": "orange", "width": 3.0, "dash": "dash"},
                    "RSI_Optimal_Support": {"color": "lightblue", "width": 3.0, "dash": "dash"}
                },
                "Monte Carlo Optimization": {
                    "MC_Resistance_Score": {"color": "darkred", "type": "line", "width": 3.0},
                    "MC_Support_Score": {"color": "darkgreen", "type": "line", "width": 3.0},
                    "RSI_Resistance_Score": {"color": "orange", "type": "line", "width": 2.0},
                    "RSI_Support_Score": {"color": "lightblue", "type": "line", "width": 2.0},
                }
            }
        }
                
        return plot_config
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Calculate candles per day dynamically based on actual timeframe
        self.CANDLES_PER_DAY = self.MINUTES_IN_DAY // timeframe_to_minutes(self.timeframe)
                
        # Initialize SwingPointDetector for each timeframe
        self.swing_detectors = {}
        for timeframe in self.TIMEFRAME_HIERARCHY:
            params = self.SWING_DETECTOR_PARAMS.get(timeframe, self.SWING_DETECTOR_PARAMS['5m']) # Default to 5m if not found
            self.swing_detectors[timeframe] = SwingPointDetector(**params)
        
        # Ensure the main timeframe (5m) is always available
        if self.timeframe not in self.swing_detectors:
            params = self.SWING_DETECTOR_PARAMS.get(self.timeframe, self.SWING_DETECTOR_PARAMS['5m'])
            self.swing_detectors[self.timeframe] = SwingPointDetector(**params)
                
        # Initialize signal generator
        self.signal_generator = SignalGenerator(self)
        
        # Initialize trendline storage for all iterations
        self.stored_trendlines = []  # List to store all Trendline objects from each Monte Carlo iteration
        
        # Initialize RSI trendline storage
        self.stored_rsi_trendlines = []  # List to store RSI-specific Trendline objects
        
        # Initialize trendline storage for each timeframe
        self.stored_trendlines_5m = []
        self.stored_trendlines_1h = []
        self.stored_trendlines_4h = []
        self.stored_trendlines_1d = []
        self.stored_trendlines_1w = []
        self.stored_trendlines_2w = []
        
        # Initialize heartbeat tracking for live trading mode
        self.last_recalculation_time = None  # Track when we last recalculated
        self.backtest_executed = False  # Track if Monte Carlo has been executed at least once
        self.rsi_backtest_executed = False  # Track if RSI Monte Carlo has been executed at least once
        
        print(f"RiskMetrics strategy initialized with Monte Carlo architecture:")
        print(f"  Lookback periods: Fixed periods for Monte Carlo optimization")
        print(f"  Monte Carlo iterations: {self.MC_ITERATIONS}")
        print(f"  Signal generator: Initialized for modular signal generation")
        print(f"  Monte Carlo functions: Using standalone functions from trendline.py")
        print(f"  Rolling Monte Carlo optimization: {'ENABLED' if self.enable_rolling_mc_optimization.value else 'DISABLED'}")
        print(f"  RSI Trendlines: {'ENABLED' if self.enable_rsi_trendlines.value else 'DISABLED'}")
        print(f"  Swing Point Detector: Initialized for swing point detection")
        if self.enable_rolling_mc_optimization.value:
            print(f"    - Eliminates lookahead bias for realistic backtesting")
            print(f"    - Uses {self.mc_lookback_window_candles.value} candle lookback window")
            print(f"    - Recalculates every {self.mc_recalc_interval_minutes.value} minutes")
        else:
            print(f"    - Using original method (faster but with potential lookahead bias)")

    @staticmethod
    def extract_latest_bounce_timestamp(trendline: Trendline) -> Optional[pd.Timestamp]:
        """
        Extract the most recent bounce timestamp from a single trendline object.

        Args:
            trendline: A Trendline object

        Returns:
            pd.Timestamp or None: The most recent bounce timestamp, or None if no bounces found
        """
        if hasattr(trendline, 'bounce_timestamps') and trendline.bounce_timestamps:
            return max(trendline.bounce_timestamps)
        return None


    def informative_pairs(self):
        """
        Define informative pair mappings for the strategy.
        Defines pairs and timeframes to be cached for the strategy.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        # Add each pair with required timeframes
        for pair in pairs:
            informative_pairs += [
                (pair, '1h'),   # 1h timeframe analysis
                (pair, '4h'),   # 4h timeframe analysis
                (pair, '1d'),   # 1d timeframe analysis
                (pair, '1w'),   # 1w timeframe analysis
            ]
            
        return informative_pairs

    def _execute_fractal_analysis(self, dataframe: DataFrame, metadata: dict, timeframe: str) -> DataFrame:
        """
        Execute fractal analysis for a given timeframe using the 12-24 month approach.
        Only uses Monte Carlo optimization on recent data (no parent timeframe dependency).
        
        Args:
            dataframe: DataFrame to analyze (already filtered to 12-24 month window)
            metadata: Strategy metadata
            timeframe: Current timeframe being analyzed (e.g., '1w', '1d', '4h', '1h')
            
        Returns:
            DataFrame: Updated dataframe with analysis results
        """
        if dataframe is None or len(dataframe) == 0:
            print(f"Warning: Empty dataframe provided for fractal analysis on {timeframe}")
            return dataframe
            
        print(f"=== ANALYSE {timeframe.upper()} POUR {metadata['pair']} (12-24 MONTH APPROACH) ===")

        # Initialize analysis environment
        self._prepare_fractal_analysis_environment(dataframe, timeframe)
        
        # Execute Monte Carlo optimization on the recent data window
        print(f"Executing Monte Carlo optimization for resistance and support on {len(dataframe)} recent candles")
        
        optimization_results = monte_carlo_period_optimization(
            dataframe, metadata['pair'],
            mc_iterations=self.MC_ITERATIONS,
            trendline_proximity_threshold=self._get_trendline_proximity_threshold(timeframe)
        )
        
        if optimization_results:
            # Process resistance results
            self._process_monte_carlo_results(
                dataframe, timeframe, 'resistance', 'resistance', optimization_results
            )
            
            # Process support results  
            self._process_monte_carlo_results(
                dataframe, timeframe, 'support', 'support', optimization_results
            )
            
            print(f"Monte Carlo optimization completed for {timeframe}")
        else:
            print(f"Monte Carlo optimization failed for {timeframe}")
                
        return dataframe

    def _prepare_fractal_analysis_environment(self, dataframe: DataFrame, timeframe: str) -> None:
        """
        Prepare the environment for fractal analysis.
        
        Args:
            dataframe: DataFrame to prepare
            timeframe: Current timeframe being analyzed
        """
        # Clear stored trendlines for this timeframe
        timeframe_storage_attr = f"stored_trendlines_{timeframe}"
        if hasattr(self, timeframe_storage_attr):
            getattr(self, timeframe_storage_attr).clear()
        
        # Initialize dataframe columns and detect swing points
        self._initialize_dataframe_columns(dataframe)
        # Use the appropriate SwingPointDetector for this timeframe
        detector = self.swing_detectors.get(timeframe, self.swing_detectors.get('5m'))
        detector.find_and_map_swing_points(dataframe)

    def _process_monte_carlo_results(self, dataframe: DataFrame, timeframe: str,
                                   analysis_type: str, column_suffix: str, 
                                   optimization_results: Dict) -> None:
        """
        Process Monte Carlo optimization results.
        
        Args:
            dataframe: DataFrame to update
            timeframe: Current timeframe
            analysis_type: Type of analysis
            column_suffix: Suffix for dataframe column names
            optimization_results: Results from Monte Carlo optimization
        """
        trendline_key = f'best_{analysis_type}_trendline'
        trendline = optimization_results.get(trendline_key)
        
        if trendline:
            self._store_trendline_and_update_dataframe(
                dataframe, timeframe, trendline, column_suffix
            )

    def _store_trendline_and_update_dataframe(self, dataframe: DataFrame, timeframe: str,
                                            trendline: Trendline, column_suffix: str) -> None:
        """
        Store trendline and update dataframe with latest bounce information.
        
        Args:
            dataframe: DataFrame to update
            timeframe: Current timeframe
            trendline: Trendline object to store
            column_suffix: Suffix for dataframe column names
        """
        # Store trendline in appropriate timeframe storage
        timeframe_storage_attr = f"stored_trendlines_{timeframe}"
        if hasattr(self, timeframe_storage_attr):
            trendline.source_timeframe = timeframe
            getattr(self, timeframe_storage_attr).append(trendline)
        
        # Update dataframe with latest bounce timestamp
        latest_bounce = self.extract_latest_bounce_timestamp(trendline)
        dataframe[f'latest_bounce_{column_suffix}'] = latest_bounce

    def _get_trendline_proximity_threshold(self, timeframe: str = None) -> float:
        """
        Get the trendline proximity threshold for a specific timeframe or the current strategy timeframe.
        
        Args:
            timeframe: Specific timeframe to get threshold for (optional)
            
        Returns:
            float: Proximity threshold value for the timeframe
        """
        target_timeframe = timeframe if timeframe else self.timeframe
        return self.TRENDLINE_PROXIMITY_THRESHOLDS.get(target_timeframe, 0.005)  # Default to 0.5%

    def _get_and_analyze_timeframe_data(self, pair: str, timeframe: str, 
                                      dataframe: DataFrame) -> DataFrame:
        """
        Orchestrate the complete timeframe analysis process using the 12-24 month approach.
        
        This method coordinates the workflow:
        1. Retrieve timeframe data
        2. Filter to 12-24 month window
        3. Execute fractal analysis on recent data only
        4. Merge results back to main dataframe
        
        Args:
            pair: Trading pair
            timeframe: Timeframe to analyze
            dataframe: Original dataframe for merging
            
        Returns:
            DataFrame: Analyzed and merged dataframe
        """
        if not self.dp:
            return dataframe
            
        # Step 1: Retrieve timeframe data
        informative_dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative_dataframe is None or len(informative_dataframe) == 0:
            print(f"Warning: No data available for {pair} on {timeframe} timeframe")
            return dataframe
            
        self._set_timeframe_start_date(timeframe, informative_dataframe)
        
        # Step 2: Filter to 12-24 month window (use 18 months as optimal balance)
        filtered_dataframe = self._filter_to_recent_window(informative_dataframe, months=18)
        if len(filtered_dataframe) == 0:
            print(f"Warning: No data available after filtering for {pair} on {timeframe} timeframe")
            return dataframe
        
        # Step 3: Execute fractal analysis on filtered data only
        analyzed_dataframe = self._execute_fractal_analysis(
            filtered_dataframe, {'pair': pair}, timeframe
        )
        if analyzed_dataframe is None or len(analyzed_dataframe) == 0:
            print(f"Warning: Fractal analysis produced no results for {pair} on {timeframe} timeframe")
            return dataframe
        
        # Step 4: Merge results back to main dataframe
        merged_dataframe = merge_informative_pair(
            dataframe, analyzed_dataframe, self.timeframe, timeframe, ffill=True
        )
        
        return merged_dataframe

    def _filter_to_recent_window(self, dataframe: DataFrame, months: int = 18) -> DataFrame:
        """
        Filter dataframe to only include data from the last N months.
        
        Args:
            dataframe: DataFrame to filter
            months: Number of months to keep (default: 18)
            
        Returns:
            DataFrame: Filtered dataframe with only recent data
        """
        if len(dataframe) == 0:
            return dataframe
            
        # Get the last date in the dataframe
        last_date = dataframe['date'].iloc[-1]
        
        # Calculate the cutoff date (N months ago)
        cutoff_date = last_date - pd.DateOffset(months=months)
        
        # Filter dataframe to only include data after cutoff
        recent_dataframe = dataframe[dataframe['date'] >= cutoff_date].copy()
        
        print(f"Filtered dataframe from {cutoff_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        print(f"Original: {len(dataframe)} candles, Filtered: {len(recent_dataframe)} candles ({len(recent_dataframe)/len(dataframe)*100:.1f}%)")
        
        return recent_dataframe

    def _set_timeframe_start_date(self, timeframe: str, dataframe: DataFrame) -> None:
        """
        Set the start date for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            dataframe: Dataframe containing the timeframe data
        """
        start_date_attr = f"timeframe_{timeframe}_start_date"
        if len(dataframe) > 0:
            start_date = dataframe['date'].iloc[0]
            setattr(self, start_date_attr, start_date)
            print(f"Date de la première ligne du dataframe {timeframe}: {start_date}")
        else:
            print(f"Warning: Empty dataframe for timeframe {timeframe}")
            setattr(self, start_date_attr, None)

    def _create_rsi_dataframe_for_trendlines(self, dataframe: DataFrame) -> DataFrame:
        """
        Create a dataframe suitable for RSI trendline analysis.
        Maps RSI swing points to OHLCV format for trendline analysis.
        
        Args:
            dataframe: Original dataframe with RSI data and swing points
            
        Returns:
            DataFrame: RSI dataframe with swing points mapped to OHLCV format
        """
        if 'rsi' not in dataframe.columns:
            return None
            
        # Create RSI dataframe with OHLCV format using RSI values
        rsi_df = dataframe[['date', 'rsi']].copy()
        rsi_df.rename(columns={'rsi': 'close'}, inplace=True)
        
        # For RSI trendlines, we use RSI value as all OHLCV components
        rsi_df['open'] = rsi_df['close']
        rsi_df['high'] = rsi_df['close']
        rsi_df['low'] = rsi_df['close']
        rsi_df['volume'] = 1.0  # Dummy volume
        
        # Map RSI swing points to highs/lows columns
        rsi_df['all_highs'] = dataframe.get('all_highs_rsi', np.nan)
        rsi_df['all_lows'] = dataframe.get('all_lows_rsi', np.nan)
        
        return rsi_df

    def _execute_rolling_rsi_monte_carlo_recalculation(self, dataframe: DataFrame, i: int, metadata: dict) -> Optional[Tuple[Trendline, Trendline]]:
        """
        Execute RSI Monte Carlo recalculation for a specific candle index.
        
        Args:
            dataframe: The full dataframe
            i: Current candle index
            metadata: Strategy metadata
            
        Returns:
            Tuple of (best_resistance_trendline, best_support_trendline) or None if failed
        """
        # Create RSI dataframe for trendline analysis
        rsi_dataframe = self._create_rsi_dataframe_for_trendlines(dataframe)
        if rsi_dataframe is None:
            return None
        
        # Define the lookback window for the current candle (point-in-time data only)
        lookback_start = max(0, i - self.mc_lookback_window_candles.value)
        current_rsi_slice = rsi_dataframe.iloc[lookback_start:i].copy()

        print(f"  RSI analysis using data slice: {lookback_start} to {i} ({len(current_rsi_slice)} candles)")
        
        # Execute Monte Carlo optimization on RSI data
        optimization_results = monte_carlo_period_optimization(
            current_rsi_slice, f"{metadata['pair']}_RSI", self.mc_recalc_interval_minutes.value,
            self.MC_ITERATIONS, self.rsi_trendline_proximity_threshold.value
        )
        
        # Extract trendline objects directly
        best_resistance_trendline = optimization_results.get('best_resistance_trendline')
        best_support_trendline = optimization_results.get('best_support_trendline')
        
        # Mark trendlines as RSI-specific
        if best_resistance_trendline:
            best_resistance_trendline.trendline_type = 'rsi_resistance'
        if best_support_trendline:
            best_support_trendline.trendline_type = 'rsi_support'
        
        return (best_resistance_trendline, best_support_trendline) if (best_resistance_trendline or best_support_trendline) else None

    def _execute_rolling_rsi_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute rolling Monte Carlo optimization for RSI trendlines to eliminate lookahead bias.
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        print("=== EXECUTING RSI Rolling Monte Carlo Optimization ===")
        
        # Initialize optimization parameters
        recalc_interval_candles = self.mc_recalc_interval_minutes.value // timeframe_to_minutes(self.timeframe)
        min_required_candles = max(self.MIN_LOOKBACK_PERIOD, 100)
        
        # Initialize variables used in the loop
        rsi_resistance_trendline = None
        rsi_support_trendline = None

        # Main rolling optimization loop for RSI
        for i in range(min_required_candles, len(dataframe)):
            current_time = dataframe['date'].iloc[i]
            should_recalculate = self._should_recalculate_at_candle(i, min_required_candles, recalc_interval_candles)

            if should_recalculate:
                print(f"RSI Monte Carlo recalculation at index {i} (time: {current_time})")
                
                trendline_results = self._execute_rolling_rsi_monte_carlo_recalculation(dataframe, i, metadata)
                
                if trendline_results:
                    rsi_resistance_trendline, rsi_support_trendline = trendline_results
                    
                    # Store RSI trendline objects from this iteration
                    if rsi_resistance_trendline:
                        self.stored_rsi_trendlines.append(rsi_resistance_trendline)
                    
                    if rsi_support_trendline:
                        self.stored_rsi_trendlines.append(rsi_support_trendline)

            # Project RSI trendlines forward using slopes
            self._project_rsi_trendlines_forward(
                dataframe, i, rsi_resistance_trendline, rsi_support_trendline
            )

    def _project_rsi_trendlines_forward(self, dataframe: DataFrame, current_index: int, 
                                      rsi_resistance_trendline: Optional[Trendline], 
                                      rsi_support_trendline: Optional[Trendline]) -> None:
        """
        Project RSI trendlines forward and update dataframe columns.
        
        Args:
            dataframe: DataFrame to update
            current_index: Current candle index
            rsi_resistance_trendline: Current RSI resistance trendline
            rsi_support_trendline: Current RSI support trendline
        """
        current_time = dataframe['date'].iloc[current_index]
        
        # Project RSI resistance trendline
        if rsi_resistance_trendline:
            projected_resistance = rsi_resistance_trendline.get_price_at_time(
                current_time, timeframe_to_minutes(self.timeframe)
            )
            if projected_resistance is not None:
                dataframe.iloc[current_index, dataframe.columns.get_loc('RSI_Optimal_Resistance')] = projected_resistance
                dataframe.iloc[current_index, dataframe.columns.get_loc('RSI_Resistance_Score')] = float(rsi_resistance_trendline.bounce_count)
        
        # Project RSI support trendline
        if rsi_support_trendline:
            projected_support = rsi_support_trendline.get_price_at_time(
                current_time, timeframe_to_minutes(self.timeframe)
            )
            if projected_support is not None:
                dataframe.iloc[current_index, dataframe.columns.get_loc('RSI_Optimal_Support')] = projected_support
                dataframe.iloc[current_index, dataframe.columns.get_loc('RSI_Support_Score')] = float(rsi_support_trendline.bounce_count)

    def _execute_non_rolling_rsi_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute non-rolling RSI Monte Carlo optimization (original method with potential lookahead bias).
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        print("=== Using Original RSI Monte Carlo Optimization (with potential lookahead bias) ===")

        # Create RSI dataframe for trendline analysis
        rsi_dataframe = self._create_rsi_dataframe_for_trendlines(dataframe)
        if rsi_dataframe is None:
            print("No RSI data available for trendline analysis")
            return

        # Execute Monte Carlo optimization on RSI data
        optimization_results = monte_carlo_period_optimization(
            rsi_dataframe, f"{metadata['pair']}_RSI",
            mc_iterations=self.MC_ITERATIONS,
            trendline_proximity_threshold=self.rsi_trendline_proximity_threshold.value
        )
        
        if optimization_results:
            # Process RSI resistance results
            rsi_resistance_trendline = optimization_results.get('best_resistance_trendline')
            if rsi_resistance_trendline:
                rsi_resistance_trendline.trendline_type = 'rsi_resistance'
                self.stored_rsi_trendlines.append(rsi_resistance_trendline)
                self._apply_rsi_trendline_to_dataframe(dataframe, rsi_resistance_trendline, 'resistance')
            
            # Process RSI support results  
            rsi_support_trendline = optimization_results.get('best_support_trendline')
            if rsi_support_trendline:
                rsi_support_trendline.trendline_type = 'rsi_support'
                self.stored_rsi_trendlines.append(rsi_support_trendline)
                self._apply_rsi_trendline_to_dataframe(dataframe, rsi_support_trendline, 'support')
                
            print(f"RSI Monte Carlo optimization completed for {metadata['pair']}")
        else:
            print(f"RSI Monte Carlo results not yet available for {metadata['pair']}.")

    def _apply_rsi_trendline_to_dataframe(self, dataframe: DataFrame, trendline: Trendline, trendline_type: str) -> None:
        """
        Apply a single RSI trendline to the dataframe by calculating its price at each timestamp.
        
        Args:
            dataframe: DataFrame to populate
            trendline: Trendline object to apply
            trendline_type: Either 'resistance' or 'support'
        """
        # Create boolean mask for active timestamps (vectorized)
        active_mask = dataframe['date'].apply(lambda ts: trendline.is_active_at_time(ts))

        # Calculate prices for all active timestamps at once (vectorized)
        active_timestamps = dataframe.loc[active_mask, 'date']
        prices = active_timestamps.apply(lambda ts: trendline.get_price_at_time(ts, timeframe_to_minutes(self.timeframe)))
        
        # Apply to appropriate RSI columns using vectorized assignment
        if trendline_type == 'resistance':
            dataframe.loc[active_mask, 'RSI_Optimal_Resistance'] = prices
            dataframe.loc[active_mask, 'RSI_Resistance_Score'] = float(trendline.bounce_count)
        elif trendline_type == 'support':
            dataframe.loc[active_mask, 'RSI_Optimal_Support'] = prices
            dataframe.loc[active_mask, 'RSI_Support_Score'] = float(trendline.bounce_count)

    def _populate_rsi_from_existing_trendlines(self, dataframe: DataFrame, metadata: dict) -> bool:
        """
        Draw every stored RSI trendline during their exact start/end time periods.
        
        Args:
            dataframe: DataFrame to populate with existing RSI trendline data
            metadata: Strategy metadata containing pair information
            
        Returns:
            bool: True if any RSI trendlines were drawn, False otherwise
        """
        if not self.stored_rsi_trendlines:
            return False
        
        pair = metadata.get('pair', 'UNKNOWN')
        print(f"Drawing {len(self.stored_rsi_trendlines)} stored RSI trendlines for {pair}")
                
        # Use vectorized operations instead of nested loops
        for trendline in self.stored_rsi_trendlines:
            # Create boolean mask for active timestamps (vectorized)
            active_mask = dataframe['date'].apply(lambda ts: trendline.is_active_at_time(ts))
            
            if not active_mask.any():
                continue  # Skip if no active timestamps

            if trendline.trendline_type == 'rsi_resistance':
                self._apply_rsi_trendline_to_dataframe(dataframe, trendline, 'resistance')
            elif trendline.trendline_type == 'rsi_support':
                self._apply_rsi_trendline_to_dataframe(dataframe, trendline, 'support')

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame.
        
        Dual-mode operation:
        1. Initial launch (backtest): Rolling Monte Carlo on all historical data
        2. Live trading (heartbeat): Non-rolling Monte Carlo only when recalc interval is reached
        """
        print("Populating indicators", dataframe)
        if len(dataframe) == 0 or not self.dp:
            return dataframe
        
        pair = metadata['pair']
        
        # Calculate Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        
        # Calculate Bollinger Band tags
        dataframe['bb_high_tag'] = np.nan
        dataframe['bb_low_tag'] = np.nan
        
        # High tag when price touches or crosses upper band
        high_touch_mask = (dataframe['high'] >= dataframe['bb_upperband'])
        dataframe.loc[high_touch_mask, 'bb_high_tag'] = dataframe.loc[high_touch_mask, 'high']
        
        # Low tag when price touches or crosses lower band
        low_touch_mask = (dataframe['low'] <= dataframe['bb_lowerband'])
        dataframe.loc[low_touch_mask, 'bb_low_tag'] = dataframe.loc[low_touch_mask, 'low']
        
        # Define timeframes to analyze independently (no dependency chain - 12-24 month approach)
        timeframes_to_analyze = ['1w', '1d', '4h', '1h']
        
        # Analyze each timeframe independently using 12-24 month window
        for timeframe in timeframes_to_analyze:
            print(f"Analyzing {timeframe} independently with 12-24 month approach")
            
            # Analyze current timeframe using only recent data
            dataframe = self._get_and_analyze_timeframe_data(
                pair, timeframe, dataframe
            )
        
        # Multiple criteria for backtest detection:
        # 1. No stored trendlines yet (first run)
        # 2. Monte Carlo has never been executed (safety check)
        is_backtest_mode = (
            len(self.stored_trendlines) == 0 or  # No trendlines stored yet (first run)
            not self.backtest_executed  # Monte Carlo has never been executed
        )
        
        # RSI backtest detection
        is_rsi_backtest_mode = (
            len(self.stored_rsi_trendlines) == 0 or  # No RSI trendlines stored yet (first run)
            not self.rsi_backtest_executed  # RSI Monte Carlo has never been executed
        )
        
        latest_candle_time = dataframe['date'].iloc[-1]

        print(f"=== MODE DETECTION ===")
        print(f"Latest candle: {latest_candle_time}")
        print(f"Dataframe length: {len(dataframe)} candles")
        print(f"Stored trendlines: {len(self.stored_trendlines)}")
        print(f"Stored RSI trendlines: {len(self.stored_rsi_trendlines)}")
        print(f"Monte Carlo executed: {self.backtest_executed}")
        print(f"RSI Monte Carlo executed: {self.rsi_backtest_executed}")
        print(f"Price trendlines mode: {'BACKTEST' if is_backtest_mode else 'LIVE TRADING'}")
        print(f"RSI trendlines mode: {'BACKTEST' if is_rsi_backtest_mode else 'LIVE TRADING'}")
        
        # Clear stored trendlines from previous runs only in backtest mode
        if is_backtest_mode:
            self.stored_trendlines.clear()
            print(f"Cleared previous price trendline storage for backtest analysis of {metadata.get('pair', 'UNKNOWN')}")
        else:
            print(f"Keeping existing price trendlines for live trading of {metadata.get('pair', 'UNKNOWN')} ({len(self.stored_trendlines)} stored)")
            
        if is_rsi_backtest_mode:
            self.stored_rsi_trendlines.clear()
            print(f"Cleared previous RSI trendline storage for backtest analysis of {metadata.get('pair', 'UNKNOWN')}")
        else:
            print(f"Keeping existing RSI trendlines for live trading of {metadata.get('pair', 'UNKNOWN')} ({len(self.stored_rsi_trendlines)} stored)")

        # Initialize all required dataframe columns
        self._initialize_dataframe_columns(dataframe)

        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_timeperiod.value)

        # Find and map RSI swing points
        self._find_rsi_swing_points(dataframe)

        # Find and map swing points using SwingPointDetector for the main timeframe (5m)
        main_detector = self.swing_detectors.get(self.timeframe)
        main_detector.find_and_map_swing_points(dataframe)
        
        # === PRICE TRENDLINES PROCESSING ===
        if not self.backtest_executed:
            # === BACKTEST MODE: Rolling Monte Carlo ===
            print("=== EXECUTING PRICE TRENDLINES BACKTEST MODE ===")
            
            # In backtest mode, always run Monte Carlo optimization since we cleared stored trendlines
            # Execute rolling Monte Carlo optimization for backtest
            if self.enable_rolling_mc_optimization.value:
                self._execute_rolling_monte_carlo_optimization(dataframe, metadata)
            else:
                self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)

            self._update_last_recalculation_time(latest_candle_time)

            # Mark that Monte Carlo has been executed
            self.backtest_executed = True
        else:
            # === LIVE TRADING MODE: Heartbeat-based Monte Carlo ===
            print("=== EXECUTING PRICE TRENDLINES LIVE TRADING MODE ===")
            
            self._populate_from_existing_trendlines(dataframe, metadata)

            # Check if we need to recalculate based on time interval            
            if self._should_recalculate_for_heartbeat(latest_candle_time):
                print(f"Recalculation interval reached - executing non-rolling Monte Carlo")
                self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)
                # Update last recalculation time
                self._update_last_recalculation_time(latest_candle_time)

        # === RSI TRENDLINES PROCESSING ===
        if self.enable_rsi_trendlines.value:
            if not self.rsi_backtest_executed:
                # === RSI BACKTEST MODE: Rolling Monte Carlo ===
                print("=== EXECUTING RSI TRENDLINES BACKTEST MODE ===")
                
                # In backtest mode, always run RSI Monte Carlo optimization since we cleared stored RSI trendlines
                if self.enable_rolling_mc_optimization.value:
                    self._execute_rolling_rsi_monte_carlo_optimization(dataframe, metadata)
                else:
                    self._execute_non_rolling_rsi_monte_carlo_optimization(dataframe, metadata)

                # Mark that RSI Monte Carlo has been executed
                self.rsi_backtest_executed = True
            else:
                # === RSI LIVE TRADING MODE: Heartbeat-based Monte Carlo ===
                print("=== EXECUTING RSI TRENDLINES LIVE TRADING MODE ===")
                
                self._populate_rsi_from_existing_trendlines(dataframe, metadata)

                # Check if we need to recalculate RSI trendlines based on time interval            
                if self._should_recalculate_for_heartbeat(latest_candle_time):
                    print(f"RSI recalculation interval reached - executing non-rolling RSI Monte Carlo")
                    self._execute_non_rolling_rsi_monte_carlo_optimization(dataframe, metadata)

        # === Output All Stored Trendlines ===
        output_trendlines_info(self.stored_trendlines_1w)
        output_trendlines_info(self.stored_trendlines_1d)
        output_trendlines_info(self.stored_trendlines_4h)
        output_trendlines_info(self.stored_trendlines_1h)
        
        # Print latest bounce timestamps for each timeframe
        for timeframe in timeframes_to_analyze:
            resistance_col = f'latest_bounce_resistance_{timeframe}'
            support_col = f'latest_bounce_support_{timeframe}'
            
            resistance_time = dataframe[resistance_col].iloc[-1] if resistance_col in dataframe.columns else None
            support_time = dataframe[support_col].iloc[-1] if support_col in dataframe.columns else None
            
            print(f"Latest bounce timestamp {timeframe} resistance: {resistance_time}")
            print(f"Latest bounce timestamp {timeframe} support: {support_time}")

        # Print RSI trendline information
        if self.enable_rsi_trendlines.value:
            print(f"RSI trendlines stored: {len(self.stored_rsi_trendlines)}")
            if self.stored_rsi_trendlines:
                print("=== RSI TRENDLINES INFO ===")
                output_trendlines_info(self.stored_rsi_trendlines)

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
        
        # Initialize RSI trendline columns
        dataframe.loc[:, 'RSI_Optimal_Resistance'] = np.nan
        dataframe.loc[:, 'RSI_Optimal_Support'] = np.nan
        dataframe.loc[:, 'RSI_Resistance_Score'] = 0.0
        dataframe.loc[:, 'RSI_Support_Score'] = 0.0
                
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
                
        # Use vectorized operations instead of nested loops
        for trendline in self.stored_trendlines:
            # Create boolean mask for active timestamps (vectorized)
            active_mask = dataframe['date'].apply(lambda ts: trendline.is_active_at_time(ts))
            
            if not active_mask.any():
                continue  # Skip if no active timestamps

            self._apply_trendline_to_dataframe(dataframe, trendline, trendline.trendline_type)

        return True

    def _apply_trendline_to_dataframe(self, dataframe: DataFrame, trendline: Trendline, trendline_type: str) -> None:
        """
        Apply a single trendline to the dataframe by calculating its price at each timestamp.
        
        Args:
            dataframe: DataFrame to populate
            trendline: Trendline object to apply
            trendline_type: Either 'resistance' or 'support'
        """

        # Create boolean mask for active timestamps (vectorized)
        active_mask = dataframe['date'].apply(lambda ts: trendline.is_active_at_time(ts))

        # Calculate prices for all active timestamps at once (vectorized)
        active_timestamps = dataframe.loc[active_mask, 'date']
        prices = active_timestamps.apply(lambda ts: trendline.get_price_at_time(ts, timeframe_to_minutes(self.timeframe)))
        
        # Apply to appropriate columns using vectorized assignment
        if trendline.trendline_type == 'resistance':
            dataframe.loc[active_mask, 'MC_Optimal_Resistance'] = prices
            dataframe.loc[active_mask, 'MC_Resistance_Score'] = float(trendline.bounce_count)
        elif trendline.trendline_type == 'support':
            dataframe.loc[active_mask, 'MC_Optimal_Support'] = prices
            dataframe.loc[active_mask, 'MC_Support_Score'] = float(trendline.bounce_count)

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


    def _execute_rolling_monte_carlo_recalculation(self, dataframe: DataFrame, i: int, metadata: dict) -> Optional[Tuple[Trendline, Trendline]]:
        """
        Execute Monte Carlo recalculation for a specific candle index.
        
        Args:
            dataframe: The full dataframe
            i: Current candle index
            metadata: Strategy metadata
            
        Returns:
            Tuple of (best_resistance_trendline, best_support_trendline) or None if failed
        """
        # Define the lookback window for the current candle (point-in-time data only)
        lookback_start = max(0, i - self.mc_lookback_window_candles.value)
        current_dataframe_slice = dataframe.iloc[lookback_start:i].copy()

        # Find and map swing points for the current slice
        high_swing_points = self.swing_detectors[self.timeframe].find_swing_points(
            prices=dataframe['high'].values,
            price_type='high'
        )
        low_swing_points = self.swing_detectors[self.timeframe].find_swing_points(
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
        
        # Execute Monte Carlo optimization directly
        optimization_results = monte_carlo_period_optimization(
            current_dataframe_slice, metadata['pair'], self.mc_recalc_interval_minutes.value,
            self.MC_ITERATIONS, self._get_trendline_proximity_threshold()
        )
        
        # Extract trendline objects directly
        best_resistance_trendline = optimization_results.get('best_resistance_trendline')
        best_support_trendline = optimization_results.get('best_support_trendline')
        
        return (best_resistance_trendline, best_support_trendline) if (best_resistance_trendline or best_support_trendline) else None

    def _execute_rolling_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute rolling Monte Carlo optimization to eliminate lookahead bias.
        Skip simulations where valid trendlines already exist for the time period.
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        # Initialize optimization parameters
        recalc_interval_candles = self.mc_recalc_interval_minutes.value // timeframe_to_minutes(self.timeframe)
        min_required_candles = max(self.MIN_LOOKBACK_PERIOD, 100)
        
        # Initialize variables used in the loop
        resistance_trendline = None
        support_trendline = None

        # Main rolling optimization loop
        for i in range(min_required_candles, len(dataframe)):
            current_time = dataframe['date'].iloc[i]
            should_recalculate = self._should_recalculate_at_candle(i, min_required_candles, recalc_interval_candles)

            if should_recalculate:
                                        
                    trendline_results = self._execute_rolling_monte_carlo_recalculation(dataframe, i, metadata)
                    
                    if trendline_results:
                        resistance_trendline, support_trendline = trendline_results
                        
                        # Store trendline objects from this iteration
                        if resistance_trendline:
                            self.stored_trendlines.append(resistance_trendline)
                        
                        if support_trendline:
                            self.stored_trendlines.append(support_trendline)
                        

            # Project trendlines forward using slopes
            project_trendlines_forward(
                dataframe, i, resistance_trendline, support_trendline, timeframe_to_minutes(self.timeframe)
            )


    def _should_recalculate_at_candle(self, candle_index: int, min_required_candles: int, 
                                    recalc_interval_candles: int) -> bool:
        """Determine if Monte Carlo should be recalculated at the current candle."""
        return (candle_index == min_required_candles) or ((candle_index - min_required_candles) % recalc_interval_candles == 0)        


    def _execute_non_rolling_monte_carlo_optimization(self, dataframe: DataFrame, metadata: dict) -> None:
        """
        Execute non-rolling Monte Carlo optimization (original method with potential lookahead bias).
        
        Args:
            dataframe: The dataframe to process
            metadata: Strategy metadata containing pair information
        """
        print("=== Using Original Monte Carlo Optimization (with potential lookahead bias) ===")

        # Execute Monte Carlo optimization (returns both resistance and support results)
        optimization_results = monte_carlo_period_optimization(
            dataframe, metadata['pair'],
            mc_iterations=self.MC_ITERATIONS,
            trendline_proximity_threshold=self._get_trendline_proximity_threshold()
        )
        
        if optimization_results:
            # Process resistance results
            self._process_monte_carlo_results(
                dataframe, self.timeframe, 'resistance', 'resistance', optimization_results
            )
            
            # Process support results  
            self._process_monte_carlo_results(
                dataframe, self.timeframe, 'support', 'support', optimization_results
            )
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

    def _find_rsi_swing_points(self, dataframe: DataFrame) -> None:
        """
        Find swing high and low points in RSI and map them to dataframe columns.
        
        Args:
            dataframe: DataFrame containing RSI data
        """
        if 'rsi' not in dataframe.columns:
            return
            
        # Initialize RSI swing point columns
        dataframe.loc[:, 'all_highs_rsi'] = np.nan
        dataframe.loc[:, 'all_lows_rsi'] = np.nan
        
        # Clean RSI data by removing NaN values
        rsi_series = dataframe['rsi'].dropna()
        
        if len(rsi_series) < 10:  # Need at least 10 valid RSI values
            print("Not enough valid RSI data for swing point detection")
            return
        
        # Use SwingPointDetector for RSI with appropriate parameters
        rsi_detector = SwingPointDetector(
            distance=10,      # Minimum distance between RSI swing points
            prominence=0.10,  # 8% prominence for RSI swing points
            wlen=None,
            width=None
        )
        
        # Find RSI swing highs and lows using clean data
        rsi_highs = rsi_detector.find_swing_points(rsi_series.values, 'high')
        rsi_lows = rsi_detector.find_swing_points(rsi_series.values, 'low')
        
        print(f"RSI data range: {rsi_series.min():.2f} to {rsi_series.max():.2f}")
        print(f"RSI highs found: {len(rsi_highs)}")
        print(f"RSI lows found: {len(rsi_lows)}")
        
        # Map RSI swing highs back to original dataframe indices
        for idx, rsi_value in rsi_highs:
            # Convert from clean data index to original dataframe index
            original_idx = rsi_series.index[idx]
            dataframe.loc[original_idx, 'all_highs_rsi'] = rsi_value
        
        # Map RSI swing lows back to original dataframe indices
        for idx, rsi_value in rsi_lows:
            # Convert from clean data index to original dataframe index
            original_idx = rsi_series.index[idx]
            dataframe.loc[original_idx, 'all_lows_rsi'] = rsi_value

