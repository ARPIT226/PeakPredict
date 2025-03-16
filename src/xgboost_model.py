import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load CSV File
df = pd.read_csv("../data/stock.csv")

# Ensure Date is in DateTime format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# --------------------- Feature Engineering ---------------------
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # 10-day SMA
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()  # 10-day EMA
df['Previous_Close'] = df['Close'].shift(1)  # Previous day's close price

# Drop NaN values caused by rolling operations
df.dropna(inplace=True)

# Select features (ensure necessary columns exist)
features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'EMA_10', 'Previous_Close']
target = 'Close'

# Prepare XGBoost model for feature selection
X = df[features]
y = df[target]

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.05, max_depth=5)
xgb_model.fit(X, y)

# Get feature importance
importance = xgb_model.feature_importances_
important_features = [features[i] for i in np.argsort(importance)[-4:]]  # Select top 4 features

# Prepare data for LSTM
df_lstm = df[important_features + [target]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_lstm)

# Convert to supervised learning format
def create_sequences(data, seq_length=60):  # Increase sequence length to capture trends
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length, -1])  # Predicting 'Close'
    return np.array(X), np.array(Y)

seq_length = 60
X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

# --------------------- Build Optimized LSTM Model ---------------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
    Dropout(0.3),  # Increased dropout to reduce overfitting
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# FIX: Rescale back to original values using correct shape
dummy_test = np.zeros((y_test.shape[0], scaled_data.shape[1]))
dummy_pred = np.zeros((y_pred.shape[0], scaled_data.shape[1]))

# Place predictions in the last column (Close price)
dummy_test[:, -1] = y_test
dummy_pred[:, -1] = y_pred.flatten()

# Inverse transform
y_test_actual = scaler.inverse_transform(dummy_test)[:, -1]
y_pred_actual = scaler.inverse_transform(dummy_pred)[:, -1]

# --------------------- Smooth Predictions ---------------------
# Apply Weighted Moving Average to Reduce Volatility
window = 5
y_pred_actual = np.convolve(y_pred_actual, np.ones(window)/window, mode='same')

# Calculate differences
difference = np.abs(y_test_actual - y_pred_actual)  # Absolute difference

# ------------------ Print Results with Difference ------------------
print("\nActual vs Predicted Stock Prices:")

for i in range(10):  # Display first 10 values
    actual = y_test_actual[i]
    predicted = y_pred_actual[i]
    diff = difference[i]
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}, Difference: {diff:.2f}")

# ------------------ Plot Actual vs Predicted ------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Price", color='blue')
plt.plot(y_pred_actual, label="Predicted Price (Smoothed)", color='red')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
