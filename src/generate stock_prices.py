import numpy as np
import pandas as pd
import random

# Set a seed for reproducibility
np.random.seed(42)

# Define the number of days in the dataset
num_days = 2000  # You can increase it as needed

# Generate a date range
date_range = pd.date_range(start="2010-01-01", periods=num_days, freq="D")

# Initialize stock price and volume
initial_price = 100  # Starting stock price
price = initial_price
prices = []
volumes = []

# Generate synthetic stock price data
for _ in range(num_days):
    daily_change = np.random.normal(loc=0, scale=2)  # Simulating daily price movement
    price += daily_change
    price = max(10, price)  # Ensure price does not go negative
    
    open_price = price + np.random.uniform(-1, 1)
    high_price = open_price + np.random.uniform(0, 2)
    low_price = open_price - np.random.uniform(0, 2)
    close_price = open_price + np.random.uniform(-1, 1)
    
    volume = random.randint(100000, 5000000)  # Random trading volume

    prices.append([open_price, high_price, low_price, close_price, volume])
    volumes.append(volume)

# Create a DataFrame
df = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
df.insert(0, 'Date', date_range)  # Insert the date column

# Save to CSV
df.to_csv("stock.csv", index=False)

print("Synthetic stock price dataset generated successfully!")
