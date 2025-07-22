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
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.001, space="buy", optimize=True)
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

    # Monte Carlo optimization parameters
    enable_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)
    
    # === Periodic Monte Carlo Parameters ===
    mc_recalc_interval_minutes = IntParameter(60, 480, default=120, space="buy", optimize=False)
    mc_lookback_window_candles = IntParameter(1000, 3000, default=2000, space="buy", optimize=False)
    
    # Rolling Monte Carlo optimization (eliminates lookahead bias)
    enable_rolling_mc_optimization = BooleanParameter(default=False, space="buy", optimize=False)

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
    TIMEFRAME_HIERARCHY = ['1h', '4h', '1d', '1w']

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
                
        return plot_config
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Calculate candles per day dynamically based on actual timeframe
        self.CANDLES_PER_DAY = self.MINUTES_IN_DAY // timeframe_to_minutes(self.timeframe)
                
        self.swing_detector = SwingPointDetector()
                
        # Initialize signal generator
        self.signal_generator = SignalGenerator(self)
        
        # Initialize trendline storage for all iterations
        self.stored_trendlines = []  # List to store all Trendline objects from each Monte Carlo iteration
        
        # Initialize trendline storage for each timeframe
        self.stored_trendlines_1h = []
        self.latest_bounce_resistance_1h = None
        self.latest_bounce_support_1h = None
        self.stored_trendlines_4h = []
        self.latest_bounce_resistance_4h = None
        self.latest_bounce_support_4h = None
        self.stored_trendlines_1d = []
        self.latest_bounce_resistance_1d = None
        self.latest_bounce_support_1d = None
        self.stored_trendlines_1w = []
        self.latest_bounce_resistance_1w = None
        self.latest_bounce_support_1w = None
        self.stored_trendlines_2w = []
        self.latest_bounce_resistance_2w = None
        self.latest_bounce_support_2w = None

        self.timeframe_1h_start_date = None
        self.timeframe_4h_start_date = None
        self.timeframe_1d_start_date = None
        self.timeframe_1w_start_date = None
        
        # Initialize heartbeat tracking for live trading mode
        self.last_recalculation_time = None  # Track when we last recalculated
        self.backtest_executed = False  # Track if Monte Carlo has been executed at least once
        
        print(f"RiskMetrics strategy initialized with Monte Carlo architecture:")
        print(f"  Lookback periods: Fixed periods for Monte Carlo optimization")
        print(f"  Monte Carlo iterations: {self.MC_ITERATIONS}")
        print(f"  Signal generator: Initialized for modular signal generation")
        print(f"  Monte Carlo functions: Using standalone functions from trendline.py")
        print(f"  Rolling Monte Carlo optimization: {'ENABLED' if self.enable_rolling_mc_optimization.value else 'DISABLED'}")
        print(f"  Swing Point Detector: Initialized for swing point detection")
        if self.enable_rolling_mc_optimization.value:
            print(f"    - Eliminates lookahead bias for realistic backtesting")
            print(f"    - Uses {self.mc_lookback_window_candles.value} candle lookback window")
            print(f"    - Recalculates every {self.mc_recalc_interval_minutes.value} minutes")
        else:
            print(f"    - Using original method (faster but with potential lookahead bias)")

    def _get_parent_timeframe(self, current_timeframe: str) -> Optional[str]:
        """
        Get the parent (higher) timeframe for the given timeframe.
        
        Args:
            current_timeframe: Current timeframe (e.g., '1h')
            
        Returns:
            str or None: Parent timeframe (e.g., '4h') or None if no parent exists
        """
        try:
            current_index = self.TIMEFRAME_HIERARCHY.index(current_timeframe)
            if current_index < len(self.TIMEFRAME_HIERARCHY) - 1:
                return self.TIMEFRAME_HIERARCHY[current_index + 1]
        except ValueError:
            pass
        return None

    def _get_timeframe_start_date(self, timeframe: str) -> Optional[pd.Timestamp]:
        """
        Get the start date for a specific timeframe.
        
        Args:
            timeframe: Timeframe to get start date for
            
        Returns:
            pd.Timestamp or None: Start date for the timeframe
        """
        start_date_attr = f"timeframe_{timeframe}_start_date"
        return getattr(self, start_date_attr, None)

    def _get_timeframe_dataframe(self, pair: str, timeframe: str) -> Optional[DataFrame]:
        """
        Get the dataframe for a specific timeframe.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe to get data for
            
        Returns:
            DataFrame or None: Dataframe for the timeframe
        """
        if not self.dp:
            return None
        return self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)

    def _calculate_trendline_in_higher_timeframe(self, pair: str, timeframe: str, 
                                               start_time: pd.Timestamp, 
                                               analysis_type: str) -> Optional[Trendline]:
        """
        Calculate trendline in a higher timeframe when start_time is before current timeframe data.
        
        Args:
            pair: Trading pair
            timeframe: Current timeframe
            start_time: Start time for trendline calculation
            analysis_type: Either 'resistance' or 'support'
            
        Returns:
            Trendline or None: Generated trendline from higher timeframe
        """
        parent_timeframe = self._get_parent_timeframe(timeframe)
        if not parent_timeframe:
            print(f"No parent timeframe available for {timeframe}, using Monte Carlo fallback")
            return None
        
        print(f"Calculating {analysis_type} trendline in {parent_timeframe} for start_time {start_time}")
        
        # Get parent timeframe dataframe
        parent_dataframe = self._get_timeframe_dataframe(pair, parent_timeframe)
        if parent_dataframe is None:
            print(f"No data available for parent timeframe {parent_timeframe}")
            return None
        
        parent_start_date = self._get_timeframe_start_date(parent_timeframe)
        
        # Check if we need to go even higher
        if parent_start_date and start_time < parent_start_date:
            print(f"start_time {start_time} < parent timeframe {parent_timeframe} start_date {parent_start_date}")
            return self._calculate_trendline_in_higher_timeframe(pair, parent_timeframe, start_time, analysis_type)
        
        # Initialize and analyze parent timeframe dataframe
        self._initialize_dataframe_columns(parent_dataframe)
        self.swing_detector.find_and_map_swing_points(parent_dataframe)
        
        # Generate forward trendline in parent timeframe
        trendline = generate_forward_trendline(
            dataframe=parent_dataframe,
            start_time=start_time,
            trendline_type=analysis_type,
            mc_recalc_interval_minutes=self.mc_recalc_interval_minutes.value,
            trendline_proximity_threshold=0.005
        )
        
        if trendline:
            trendline.source_timeframe = parent_timeframe
            print(f"Successfully generated {analysis_type} trendline in {parent_timeframe}: {trendline}")
        else:
            print(f"Failed to generate {analysis_type} trendline in {parent_timeframe}")
            
        return trendline

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

    def _execute_fractal_analysis(self, dataframe: DataFrame, metadata: dict, timeframe: str, 
                                 parent_resistance_time: Optional[pd.Timestamp], 
                                 parent_support_time: Optional[pd.Timestamp]) -> DataFrame:
        """
        Execute fractal analysis for a given timeframe using parent timeframe data.
        
        Args:
            dataframe: DataFrame to analyze
            metadata: Strategy metadata
            timeframe: Current timeframe being analyzed (e.g., '1w', '1d', '4h', '1h')
            parent_resistance_time: Latest bounce resistance time from parent timeframe
            parent_support_time: Latest bounce support time from parent timeframe
            
        Returns:
            DataFrame: Updated dataframe with analysis results
        """
        print(f"=== ANALYSE {timeframe.upper()} POUR {metadata['pair']} ===")

        # Initialize analysis environment
        self._prepare_fractal_analysis_environment(dataframe, timeframe)
        
        # Get timeframe start date for validation
        timeframe_start_date = self._get_timeframe_start_date(timeframe)
        
        # Define analysis configurations
        analysis_configurations = self._create_analysis_configurations(
            parent_resistance_time, parent_support_time
        )
        
        # Execute each analysis type
        for analysis_config in analysis_configurations:
            self._execute_single_analysis_type(
                dataframe, metadata, timeframe, timeframe_start_date, analysis_config
            )
                
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
        self.swing_detector.find_and_map_swing_points(dataframe)

    def _create_analysis_configurations(self, parent_resistance_time: Optional[pd.Timestamp], 
                                     parent_support_time: Optional[pd.Timestamp]) -> List[Dict]:
        """
        Create analysis configurations for resistance and support analysis.
        
        Args:
            parent_resistance_time: Parent timeframe resistance time
            parent_support_time: Parent timeframe support time
            
        Returns:
            List[Dict]: List of analysis configurations
        """
        return [
            {
                'type': 'resistance', 
                'start_time': parent_resistance_time,
                'column_suffix': 'resistance'
            },
            {
                'type': 'support', 
                'start_time': parent_support_time,
                'column_suffix': 'support'
            }
        ]

    def _execute_single_analysis_type(self, dataframe: DataFrame, metadata: dict, 
                                    timeframe: str, timeframe_start_date: Optional[pd.Timestamp],
                                    analysis_config: Dict) -> None:
        """
        Execute analysis for a single type (resistance or support).
        
        Args:
            dataframe: DataFrame to analyze
            metadata: Strategy metadata
            timeframe: Current timeframe
            timeframe_start_date: Start date of the timeframe
            analysis_config: Analysis configuration dictionary
        """
        analysis_type = analysis_config['type']
        start_time = analysis_config['start_time']
        column_suffix = analysis_config['column_suffix']
        
        print(f"Processing {analysis_type} analysis for {timeframe}")
        
        if start_time is not None:
            self._execute_timed_analysis(
                dataframe, metadata, timeframe, timeframe_start_date, 
                analysis_type, start_time, column_suffix
            )
        else:
            print(f"Executing Monte Carlo optimization for {analysis_type} (no start_time provided)")
            
            if optimization_results := monte_carlo_period_optimization(
                dataframe, metadata['pair'],
                mc_iterations=self.MC_ITERATIONS,
                trendline_proximity_threshold=0.1
            ):
                self._process_monte_carlo_results(
                    dataframe, timeframe, analysis_type, column_suffix, optimization_results
                )


    def _execute_timed_analysis(self, dataframe: DataFrame, metadata: dict, 
                               timeframe: str, timeframe_start_date: Optional[pd.Timestamp],
                               analysis_type: str, start_time: pd.Timestamp, 
                               column_suffix: str) -> None:
        """
        Execute analysis with a specific start time.
        
        Args:
            dataframe: DataFrame to analyze
            metadata: Strategy metadata
            timeframe: Current timeframe
            timeframe_start_date: Start date of the timeframe
            analysis_type: Type of analysis ('resistance' or 'support')
            start_time: Start time for analysis
            column_suffix: Suffix for dataframe column names
        """
        if self._should_use_higher_timeframe(timeframe_start_date, start_time):
            self._execute_higher_timeframe_analysis(
                dataframe, metadata, timeframe, analysis_type, start_time, column_suffix
            )
        else:
            self._execute_current_timeframe_analysis(
                dataframe, metadata, timeframe, analysis_type, start_time, column_suffix
            )

    def _should_use_higher_timeframe(self, timeframe_start_date: Optional[pd.Timestamp], 
                                    start_time: pd.Timestamp) -> bool:
        """
        Determine if analysis should use higher timeframe data.
        
        Args:
            timeframe_start_date: Start date of current timeframe
            start_time: Analysis start time
            
        Returns:
            bool: True if higher timeframe should be used
        """
        return timeframe_start_date is not None and start_time < timeframe_start_date

    def _execute_higher_timeframe_analysis(self, dataframe: DataFrame, metadata: dict,
                                         timeframe: str, analysis_type: str, 
                                         start_time: pd.Timestamp, column_suffix: str) -> None:
        """
        Execute analysis using higher timeframe data.
        
        Args:
            dataframe: DataFrame to analyze
            metadata: Strategy metadata
            timeframe: Current timeframe
            analysis_type: Type of analysis
            start_time: Start time for analysis
            column_suffix: Suffix for dataframe column names
        """
        print(f"start_time {start_time} < timeframe_{timeframe}_start_date {self._get_timeframe_start_date(timeframe)}")
        print(f"Calculating {analysis_type} trendline in higher timeframe")
        
        trendline = self._calculate_trendline_in_higher_timeframe(
            metadata['pair'], timeframe, start_time, analysis_type
        )
        
        if trendline:
            # Get the parent timeframe since the trendline was calculated in the parent timeframe
            parent_timeframe = self._get_parent_timeframe(timeframe)
            if parent_timeframe:
                # Store the trendline in the parent timeframe storage since it was calculated there
                self._store_trendline_and_update_dataframe(
                    dataframe, parent_timeframe, trendline, column_suffix
                )
                print(f"Generated {analysis_type} trendline from higher timeframe: {trendline}")
            else:
                print(f"Failed to get parent timeframe for {timeframe}")
        else:
            print(f"Failed to generate {analysis_type} trendline from higher timeframe")

    def _execute_current_timeframe_analysis(self, dataframe: DataFrame, metadata: dict,
                                         timeframe: str, analysis_type: str, 
                                         start_time: pd.Timestamp, column_suffix: str) -> None:
        """
        Execute analysis using current timeframe data.
        
        Args:
            dataframe: DataFrame to analyze
            metadata: Strategy metadata
            timeframe: Current timeframe
            analysis_type: Type of analysis
            start_time: Start time for analysis
            column_suffix: Suffix for dataframe column names
        """
        # Filter dataframe to data after start_time
        filtered_dataframe = self._filter_dataframe_by_start_time(dataframe, start_time)
        print(f"Filtered dataframe {timeframe} from start_time: {start_time}, {len(filtered_dataframe)} candles retained.")
        
        print(f"Generating forward trendline for {analysis_type} from {start_time}")
        
        trendline = generate_forward_trendline(
            dataframe=dataframe,
            start_time=start_time,
            trendline_type=analysis_type,
            mc_recalc_interval_minutes=self.mc_recalc_interval_minutes.value,
            trendline_proximity_threshold=self._get_trendline_proximity_threshold()
        )

        if trendline:
            self._store_trendline_and_update_dataframe(
                dataframe, timeframe, trendline, column_suffix
            )
            print(f"Generated {analysis_type} trendline: {trendline}")
        else:
            print(f"No valid {analysis_type} trendline generated from {start_time}")

    def _filter_dataframe_by_start_time(self, dataframe: DataFrame, start_time: pd.Timestamp) -> DataFrame:
        """
        Filter dataframe to include only data after the start time.
        
        Args:
            dataframe: DataFrame to filter
            start_time: Start time for filtering
            
        Returns:
            DataFrame: Filtered dataframe
        """
        mask = dataframe['date'] > start_time
        return dataframe[mask].copy()


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

    def _get_trendline_proximity_threshold(self) -> float:
        """
        Get the trendline proximity threshold for forward trendline generation.
        
        Returns:
            float: Proximity threshold value
        """
        return 0.005  # 0.5% default threshold

    def _get_and_analyze_timeframe_data(self, pair: str, timeframe: str, 
                                      parent_resistance_time: Optional[pd.Timestamp],
                                      parent_support_time: Optional[pd.Timestamp],
                                      dataframe: DataFrame) -> DataFrame:
        """
        Orchestrate the complete timeframe analysis process.
        
        This method coordinates the workflow:
        1. Retrieve timeframe data
        2. Execute fractal analysis
        3. Merge results back to main dataframe
        
        Args:
            pair: Trading pair
            timeframe: Timeframe to analyze
            parent_resistance_time: Parent timeframe resistance time
            parent_support_time: Parent timeframe support time
            dataframe: Original dataframe for merging
            
        Returns:
            DataFrame: Analyzed and merged dataframe
        """
        if not self.dp:
            return dataframe
            
        # Step 1: Retrieve timeframe data
        informative_dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        self._set_timeframe_start_date(timeframe, informative_dataframe)
        
        # Step 2: Execute fractal analysis
        analyzed_dataframe = self._execute_fractal_analysis(
            informative_dataframe, {'pair': pair}, timeframe,
            parent_resistance_time, parent_support_time
        )
        
        # Step 3: Merge results back to main dataframe
        merged_dataframe = merge_informative_pair(
            dataframe, analyzed_dataframe, self.timeframe, timeframe, ffill=True
        )
        
        return merged_dataframe

    def _set_timeframe_start_date(self, timeframe: str, dataframe: DataFrame) -> None:
        """
        Set the start date for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            dataframe: Dataframe containing the timeframe data
        """
        start_date_attr = f"timeframe_{timeframe}_start_date"
        start_date = dataframe['date'].iloc[0]
        setattr(self, start_date_attr, start_date)
        print(f"Date de la première ligne du dataframe {timeframe}: {start_date}")

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
        
        # Define timeframes to analyze in order (dependency chain)
        timeframes_to_analyze = [
            {'timeframe': '1w', 'parent_resistance': None, 'parent_support': None},
            {'timeframe': '1d', 'parent_resistance': 'latest_bounce_resistance_1w', 'parent_support': 'latest_bounce_support_1w'},
            {'timeframe': '4h', 'parent_resistance': 'latest_bounce_resistance_1d', 'parent_support': 'latest_bounce_support_1d'},
            {'timeframe': '1h', 'parent_resistance': 'latest_bounce_resistance_4h', 'parent_support': 'latest_bounce_support_4h'}
        ]
        
        # Analyze each timeframe in sequence
        for analysis_config in timeframes_to_analyze:
            timeframe = analysis_config['timeframe']
            parent_resistance_col = analysis_config['parent_resistance']
            parent_support_col = analysis_config['parent_support']
            
            # Get parent times if available
            parent_resistance_time = None
            parent_support_time = None
            
            if parent_resistance_col and parent_resistance_col in dataframe.columns:
                parent_resistance_time = dataframe[parent_resistance_col].iloc[-1]
            
            if parent_support_col and parent_support_col in dataframe.columns:
                parent_support_time = dataframe[parent_support_col].iloc[-1]
            
            # Analyze current timeframe
            dataframe = self._get_and_analyze_timeframe_data(
                pair, timeframe, parent_resistance_time, parent_support_time, dataframe
            )
        
        # Multiple criteria for backtest detection:
        # 1. No stored trendlines yet (first run)
        # 2. Monte Carlo has never been executed (safety check)
        is_backtest_mode = (
            len(self.stored_trendlines) == 0 or  # No trendlines stored yet (first run)
            not self.backtest_executed  # Monte Carlo has never been executed
        )
        
        latest_candle_time = dataframe['date'].iloc[-1]

        print(f"=== MODE DETECTION ===")
        print(f"Latest candle: {latest_candle_time}")
        print(f"Dataframe length: {len(dataframe)} candles")
        print(f"Stored trendlines: {len(self.stored_trendlines)}")
        print(f"Monte Carlo executed: {self.backtest_executed}")
        print(f"Backtest criteria:")
        print(f"  - No stored trendlines: {len(self.stored_trendlines) == 0}")
        print(f"  - Monte Carlo never executed: {not self.backtest_executed}")
        print(f"Mode: {'BACKTEST (Rolling Monte Carlo)' if is_backtest_mode else 'LIVE TRADING (Heartbeat-based)'}")
        
        # Clear stored trendlines from previous runs only in backtest mode
        if is_backtest_mode:
            self.stored_trendlines.clear()
            print(f"Cleared previous trendline storage for backtest analysis of {metadata.get('pair', 'UNKNOWN')}")
        else:
            print(f"Keeping existing trendlines for live trading of {metadata.get('pair', 'UNKNOWN')} ({len(self.stored_trendlines)} stored)")
            

        # Initialize all required dataframe columns
        self._initialize_dataframe_columns(dataframe)

        # Find and map swing points to both dataframes using SwingPointDetector
        self.swing_detector.find_and_map_swing_points(dataframe)
        
        if not self.backtest_executed:
            # === BACKTEST MODE: Rolling Monte Carlo ===
            print("=== EXECUTING BACKTEST MODE ===")
            
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
            print("=== EXECUTING LIVE TRADING MODE ===")
            
            self._populate_from_existing_trendlines(dataframe, metadata)

            # Check if we need to recalculate based on time interval            
            if self._should_recalculate_for_heartbeat(latest_candle_time):
                print(f"Recalculation interval reached - executing non-rolling Monte Carlo")
                self._execute_non_rolling_monte_carlo_optimization(dataframe, metadata)
                # Update last recalculation time
                self._update_last_recalculation_time(latest_candle_time)


        # === Output All Stored Trendlines ===
        output_trendlines_info(self.stored_trendlines_1w)
        output_trendlines_info(self.stored_trendlines_1d)
        output_trendlines_info(self.stored_trendlines_4h)
        output_trendlines_info(self.stored_trendlines_1h)
        
        # Print latest bounce timestamps for each timeframe
        for analysis_config in timeframes_to_analyze:
            timeframe = analysis_config['timeframe']
            resistance_col = f'latest_bounce_resistance_{timeframe}'
            support_col = f'latest_bounce_support_{timeframe}'
            
            resistance_time = dataframe[resistance_col].iloc[-1] if resistance_col in dataframe.columns else None
            support_time = dataframe[support_col].iloc[-1] if support_col in dataframe.columns else None
            
            print(f"Latest bounce timestamp {timeframe} resistance: {resistance_time}")
            print(f"Latest bounce timestamp {timeframe} support: {support_time}")

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
        
        # Execute Monte Carlo optimization directly
        optimization_results = monte_carlo_period_optimization(
            current_dataframe_slice, metadata['pair'], self.mc_recalc_interval_minutes.value,
            self.MC_ITERATIONS, self.trendline_proximity_threshold.value
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
            trendline_proximity_threshold=self.trendline_proximity_threshold.value
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

