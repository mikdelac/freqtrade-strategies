import pandas as pd
import json
from datetime import datetime

def parse_order_book_data(file_path):
    # Load the data from SOL.txt
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Extract relevant information and organize it in a DataFrame
    records = []
    for line in data:
        try:
            entry = json.loads(line)
            # Use the 'time' field from the 'raw' section if 'data' is not present
            time = datetime.fromtimestamp(entry['raw']['data']['time'] / 1000)  # Convert milliseconds to seconds
            levels = entry['raw']['data']['levels']
            for level_group in levels:
                for order in level_group:
                    records.append({
                        'timestamp': time,
                        'price': float(order['px']),
                        'size': float(order['sz'])
                    })

        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(records)
    return df

def convert_to_kline(df, interval='1T'):
    # Check if 'timestamp' column is present
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the DataFrame")

    # Resample the data to K-line format (OHLCV)
    kline = df.set_index('timestamp').resample(interval).agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum'
    })

    # Rename columns to OHLCV
    kline.columns = ['open', 'high', 'low', 'close', 'volume']

    # Drop empty intervals
    kline.dropna(inplace=True)

    return kline

def format_kline_data(kline):
    # Convert the DataFrame to the specified format
    formatted_data = []
    for index, row in kline.iterrows():
        timestamp = int(index.timestamp() * 1000)  # Convert to milliseconds
        formatted_data.append([
            timestamp,
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ])
    return formatted_data

# Load order book data
df = parse_order_book_data('user_data/data/hyperliquid/SOL')

# Debugging: Print the first few rows of the DataFrame to check its structure
print(df.head())
print(df.tail())

# Convert to 10-minute K-line data
kline_data = convert_to_kline(df, interval='1min')

# Format the K-line data
formatted_data = format_kline_data(kline_data)

# Output the results
print(formatted_data)

# Save to CSV for further analysis or usage
kline_data.to_csv('SOL_kline.csv')
