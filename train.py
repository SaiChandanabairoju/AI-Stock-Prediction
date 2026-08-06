import os
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# Create Project Folders
# =====================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================================
# Download Stock Data
# =====================================================

STOCK = "AAPL"

print("=" * 50)
print(f"Downloading {STOCK} stock data...")
print("=" * 50)

data = yf.download(
    STOCK,
    start="2015-01-01",
    end="2025-12-31",
    auto_adjust=True
)

if data.empty:
    raise Exception("Failed to download stock data.")

# Save dataset
data.to_csv(f"dataset/{STOCK}.csv")

print("Dataset downloaded successfully!")

# =====================================================
# Extract Close Price
# =====================================================

if isinstance(data.columns, pd.MultiIndex):
    close_data = data.xs("Close", axis=1, level=0)
else:
    close_data = data[["Close"]]

print("\nFirst 5 Rows")
print(close_data.head())

# =====================================================
# Scale Data
# =====================================================

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(close_data)

joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler Saved!")

# =====================================================
# Create Sequences
# =====================================================

sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("\nShape of X :", X.shape)
print("Shape of y :", y.shape)

# =====================================================
# Split Dataset
# =====================================================

train_size = int(len(X) * 0.80)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("\nTraining Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# =====================================================
# Build LSTM Model
# =====================================================

model = Sequential()

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(sequence_length, 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=64,
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(Dense(32))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\n")
model.summary()

# =====================================================
# Train Model
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# Save Model
# =====================================================

model.save("models/lstm_model.keras")

print("\nModel Saved Successfully!")

# =====================================================
# Predict
# =====================================================

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# =====================================================
# Evaluation
# =====================================================

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

mae = mean_absolute_error(
    actual,
    predictions
)

mape = np.mean(
    np.abs((actual - predictions) / actual)
) * 100

accuracy = 100 - mape

print("\n")
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"RMSE      : {rmse:.2f}")
print(f"MAE       : {mae:.2f}")
print(f"MAPE      : {mape:.2f}%")
print(f"Accuracy  : {accuracy:.2f}%")

# =====================================================
# Plot Prediction
# =====================================================

plt.figure(figsize=(15, 6))

plt.plot(
    actual,
    label="Actual Price",
    linewidth=2
)

plt.plot(
    predictions,
    label="Predicted Price",
    linewidth=2
)

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# Plot Loss
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# Moving Average Plot
# =====================================================

close_plot = close_data.copy()

if isinstance(close_plot, pd.DataFrame):

    if close_plot.shape[1] == 1:

        close_plot.columns = ["Close"]

close_plot["MA50"] = close_plot["Close"].rolling(50).mean()
close_plot["MA200"] = close_plot["Close"].rolling(200).mean()

plt.figure(figsize=(15,6))

plt.plot(close_plot["Close"], label="Close Price")

plt.plot(close_plot["MA50"], label="50-Day MA")

plt.plot(close_plot["MA200"], label="200-Day MA")

plt.title(f"{STOCK} Moving Averages")

plt.legend()

plt.grid(True)

plt.show()

print("\nTraining Completed Successfully!")

print("\nFiles Generated:")
print("--------------------------")
print(f"dataset/{STOCK}.csv")
print("models/scaler.pkl")
print("models/lstm_model.keras")