import pandas as pd
import random
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------ Step 1: Load & Prepare Stock Dataset ------------------
stock_df = pd.read_csv("../data/stock.csv")
stock_df['Date'] = pd.to_datetime(stock_df['Date'])  # Convert Date to datetime

# ------------------ Step 2: Generate Sentiment Dataset ------------------
positive_headlines = [
    "Stock market surges as tech giants report strong earnings",
    "Investors confident as economy shows strong recovery",
    "Company X posts record-breaking revenue growth",
]

negative_headlines = [
    "Market crashes due to inflation concerns",
    "Economic downturn leads to massive sell-off",
    "Company Z faces bankruptcy, stock prices plummet",
]

neutral_headlines = [
    "Analysts predict stable growth next quarter",
    "Market closes with mixed results today",
    "Stock trading volume remains steady this week",
]

# Randomly generate headlines with corresponding dates from stock_df
num_samples = 1200
dates = np.random.choice(stock_df['Date'].values, num_samples)  # Use stock dates

headlines = []
sentiments = []

for date in dates:
    sentiment_type = random.choice(["positive", "negative", "neutral"])
    
    if sentiment_type == "positive":
        headline = random.choice(positive_headlines)
    elif sentiment_type == "negative":
        headline = random.choice(negative_headlines)
    else:
        headline = random.choice(neutral_headlines)
    
    # Sentiment Analysis using TextBlob
    sentiment_score = TextBlob(headline).sentiment.polarity
    
    headlines.append((date, headline, sentiment_score))

news_df = pd.DataFrame(headlines, columns=["Date", "headline", "sentiment_score"])
news_df['Date'] = pd.to_datetime(news_df['Date'])

# Save to CSV
news_df.to_csv("news_headlines.csv", index=False)

# ------------------ Step 3: Merge Stock & Sentiment Data ------------------
merged_df = pd.merge(stock_df, news_df, on="Date", how="left")
merged_df.fillna(0, inplace=True)  # Fill missing sentiment scores with 0

# Select features
features = ["Open", "High", "Low", "Volume", "sentiment_score"]
target = "Close"

# ------------------ Step 4: Train Hybrid XGBoost + LSTM Model ------------------
# Prepare data
X = merged_df[features]
y = merged_df[target]

# XGBoost for Feature Importance
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X, y)

# Get important features
importance = xgb_model.feature_importances_
important_features = [features[i] for i in np.argsort(importance)[-3:]]  # Top 3 features

# Prepare data for LSTM
df_lstm = merged_df[important_features + [target]]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lstm)

# Convert to LSTM sequences
def create_sequences(data, seq_length=50):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length, -1])
    return np.array(X_seq), np.array(y_seq)

seq_length = 50
X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Rescale predictions
dummy_test = np.zeros((y_test.shape[0], scaled_data.shape[1]))
dummy_pred = np.zeros((y_pred.shape[0], scaled_data.shape[1]))

dummy_test[:, -1] = y_test
dummy_pred[:, -1] = y_pred.flatten()

y_test_actual = scaler.inverse_transform(dummy_test)[:, -1]
y_pred_actual = scaler.inverse_transform(dummy_pred)[:, -1]

# ------------------ Step 5: Plot & Display Results ------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Price", color="blue")
plt.plot(y_pred_actual, label="Predicted Price", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Print results
# Calculate difference
result_df = pd.DataFrame({
    "Actual Price": y_test_actual,
    "Predicted Price": y_pred_actual,
    "Difference": y_test_actual - y_pred_actual  # Absolute difference
})

# Print first 20 rows
print(result_df.head(20))

