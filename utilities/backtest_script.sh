#!/bin/bash

# Initialize variables to track the best RSI_TIMEPERIOD and its profit
best_rsi_timeperiod=0
best_profit=-999999

# Define the range of RSI_TIMEPERIOD values
for RSI_TIMEPERIOD in {10..40}
do
  # Export the RSI_TIMEPERIOD environment variable
  export RSI_TIMEPERIOD=$RSI_TIMEPERIOD

  # Run the backtest and capture the output
  output=$(freqtrade backtesting --config user_data/config.json --timeframe 1h --timerange 20240101- --strategy SampleStrategy)

  # Extract the total profit percentage using regex
  profit=$(echo "$output" | grep -oP 'Total profit %\s+\│\s+\K-?\d+\.\d+')

  # Extract the average duration of trades using regex
  avg_duration_winners=$(echo "$output" | grep -oP 'Avg. Duration Winners\s+\│\s+\K\d+ days')
  avg_duration_losers=$(echo "$output" | grep -oP 'Avg. Duration Loser\s+\│\s+\K\d+ days')

  # Convert the average duration to integers
  avg_duration_winners=${avg_duration_winners// days/}
  avg_duration_losers=${avg_duration_losers// days/}

  # Print the RSI_TIMEPERIOD, profit, and average durations for clarity
  echo "RSI_TIMEPERIOD: $RSI_TIMEPERIOD, Total profit %: $profit, Avg. Duration Winners: $avg_duration_winners days, Avg. Duration Losers: $avg_duration_losers days"

  # Check if this profit is better than the best profit found so far and the average duration is below 14 days
  if (( $(echo "$profit > $best_profit" | bc -l) )) && (( avg_duration_winners < 14 )) && (( avg_duration_losers < 14 )); then
    best_rsi_timeperiod=$RSI_TIMEPERIOD
    best_profit=$profit
  fi

  # Print a separator for clarity in the output
  echo "--------------------------------------------------"
done

# Print the best RSI_TIMEPERIOD and its profit
echo "Best RSI_TIMEPERIOD: $best_rsi_timeperiod, Best Total profit %: $best_profit"

# Run the backtest once again for the best RSI_TIMEPERIOD and display the output
export RSI_TIMEPERIOD=$best_rsi_timeperiod
echo "Running backtest for the best RSI_TIMEPERIOD: $best_rsi_timeperiod"
freqtrade backtesting --config user_data/config.json --timeframe 1h --timerange 20240101- --strategy SampleStrategy
