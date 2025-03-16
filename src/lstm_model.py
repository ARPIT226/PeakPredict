import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd

# ****************************** DATA LOADING ******************************

# Load dataset
df = pd.read_csv("../data/stock.csv")

# Selecting only numerical features (excluding Date and Close)
features = ["Open", "High", "Low", "Volume"]  #  Only numerical inputs
target_column = "Close"  #  Predicting Close price

# Convert to numpy arrays
X = df[features].values
y = df[target_column].values.reshape(-1, 1)  # Ensure column shape

#  Proper scaling
scaler_features = MinMaxScaler()
X_scaled = scaler_features.fit_transform(X)  # Fit on full dataset before splitting

scaler_target = MinMaxScaler()
y_scaled = scaler_target.fit_transform(y)  # Fit on full dataset before splitting

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# ****************************** SEQUENCE CREATION ******************************

def create_sequences(data_X, data_y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data_X) - seq_length):
        X_seq.append(data_X[i:i + seq_length])  # Selecting features
        y_seq.append(data_y[i + seq_length])  # Selecting corresponding Close price
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 50
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Adjust dataset split
train_size = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
y_train_seq, y_test_seq = y_seq[:train_size], y_seq[train_size:]

# Shuffle training data
X_train_seq, y_train_seq = shuffle(X_train_seq, y_train_seq, random_state=42)

# ****************************** LSTM MODEL ******************************

# Define LSTM model
lstm_model = Sequential([
    Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Using Adam optimizer with learning rate scheduling
optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model with **epochs reduced to 50**
history = lstm_model.fit(
    X_train_seq, y_train_seq, 
    epochs=50, batch_size=32,  # Reduced epochs from 100 to 50
    validation_data=(X_test_seq, y_test_seq), 
    callbacks=[early_stopping]
)

# ****************************** PREDICTIONS ******************************

# Predictions
lstm_predictions = lstm_model.predict(X_test_seq)

# Convert predictions back to real Close prices
lstm_real_predictions = scaler_target.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()

# Convert actual Close prices back to real values
lstm_real_actuals = scaler_target.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# Save LSTM model
lstm_model.save("../models/lstm_model.keras")

# ****************************** VISUALIZATION ******************************

# Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(lstm_real_actuals, label="Actual Prices", color="blue")
plt.plot(lstm_real_predictions, label="Predicted Prices", color="red", linestyle='dashed')
plt.title("Actual vs. Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()

# Difference Plot (Actual - Predicted)
# plt.figure(figsize=(12, 6))
# plt.plot(lstm_real_actuals - lstm_real_predictions, label="Prediction Difference", color="purple")
# plt.axhline(y=0, color="black", linestyle="dashed")
# plt.title("Difference Between Actual and Predicted Prices")
# plt.xlabel("Time")
# plt.ylabel("Difference (Actual - Predicted)")
# plt.legend()
# plt.grid()
# plt.show()

# ****************************** TERMINAL OUTPUT ******************************

print("\n=== ACTUAL vs. PREDICTED PRICES ===\n")
print(f"{'Time':<8}{'Actual':<15}{'Predicted':<15}{'Difference':<15}")
print("="*50)

for i in range(len(lstm_real_actuals)):
    diff = abs(lstm_real_actuals[i] - lstm_real_predictions[i])
    print(f"{i:<8}{lstm_real_actuals[i]:<15.2f}{lstm_real_predictions[i]:<15.2f}{diff:<15.2f}")

print("\n=== END OF RESULTS ===\n")
