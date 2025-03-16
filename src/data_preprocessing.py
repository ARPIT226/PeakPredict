import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Selecting features & target
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'
    
    #  Separate scalers for features & target
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))  #  Separate scaler for Close price

    #  Fit & transform features & target separately
    df[features] = scaler_features.fit_transform(df[features])
    df[target] = scaler_target.fit_transform(df[[target]])  #  Now using separate scaler!

    # Return everything correctly
    return df, scaler_features, scaler_target, features, target

# Usage Example (for testing)
if __name__ == "__main__":
    df, scaler_features, scaler_target, features, target = load_and_preprocess_data("../data/stock.csv")
    print(df.head(5))
