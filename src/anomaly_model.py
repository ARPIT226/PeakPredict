import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load CSV file
file_path = "../data/stock.csv"  # Update if needed
data = pd.read_csv(file_path)

# Selecting features (Update column names if needed)
features = ['Open', 'Close', 'High', 'Low', 'Volume']
X = data[features]

# Normalize data (Scaling for better anomaly detection)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

### *Isolation Forest Model*
contamination = 0.02  # Lower contamination for better precision
iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
iso_forest.fit(X_train)
y_pred_iso = iso_forest.predict(X_scaled)  # Apply to full dataset

# Convert predictions (1 = normal, -1 = anomaly → Change to 0 and 1)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)

### *Local Outlier Factor (LOF) Model*
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
y_pred_lof = lof.fit_predict(X_scaled)
y_pred_lof = np.where(y_pred_lof == 1, 0, 1)

# Print evaluation metrics (if ground truth is available)
if 'Class' in data.columns:
    y_true = data['Class']  # Ensure alignment with full dataset
    
    print("### Isolation Forest Performance ###")
    print(classification_report(y_true, y_pred_iso))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred_iso))

    print("\n### Local Outlier Factor Performance ###")
    print(classification_report(y_true, y_pred_lof))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred_lof))

# Visualization (Scatter plot for Open Price vs Close Price)
plt.figure(figsize=(10, 6))
plt.scatter(data['Open'], data['Close'], c=y_pred_iso, cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Anomaly (1 = anomaly, 0 = normal)")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Refined Anomaly Detection in Stock Prices")
plt.show()