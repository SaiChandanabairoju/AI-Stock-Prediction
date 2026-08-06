import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==============================
# Load Model and Scaler
# ==============================

model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler.pkl")


# ==============================
# Prediction Function
# ==============================

def predict_stock(stock):

    try:

        ticker = yf.Ticker(stock)

        data = ticker.history(period="5y", auto_adjust=True)

        if data.empty:
            return None

        # -----------------------------
        # Company Information
        # -----------------------------

        try:
            info = ticker.info
        except:
            info = {}

        company = {
            "Name": info.get("longName", stock),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }

        # -----------------------------
        # Closing Price
        # -----------------------------

        close = data["Close"].values.reshape(-1, 1)

        scaled = scaler.transform(close)

        X = []
        y = []

        sequence = 60

        for i in range(sequence, len(scaled)):
            X.append(scaled[i-sequence:i])
            y.append(scaled[i])

        X = np.array(X)
        y = np.array(y)

        # -----------------------------
        # Historical Prediction
        # -----------------------------

        predictions = model.predict(X, verbose=0)

        predictions = scaler.inverse_transform(predictions)

        actual = scaler.inverse_transform(y)

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

        # -----------------------------
        # Tomorrow Prediction
        # -----------------------------

        last60 = scaled[-60:]

        X_future = np.array([last60])

        tomorrow = model.predict(
            X_future,
            verbose=0
        )

        tomorrow_price = scaler.inverse_transform(
            tomorrow
        )[0][0]

        # -----------------------------
        # Next 7 Days
        # -----------------------------

        future = []

        current = last60.copy()

        for _ in range(7):

            pred = model.predict(
                np.array([current]),
                verbose=0
            )

            price = scaler.inverse_transform(pred)[0][0]

            future.append(price)

            current = np.vstack(
                (current[1:], pred)
            )

        # -----------------------------
        # Moving Averages
        # -----------------------------

        data["MA20"] = data["Close"].rolling(20).mean()

        data["MA50"] = data["Close"].rolling(50).mean()

        data["MA100"] = data["Close"].rolling(100).mean()

        data["MA200"] = data["Close"].rolling(200).mean()

        return {

            "data": data,

            "today_price": float(
                data["Close"].iloc[-1]
            ),

            "tomorrow_price": float(
                tomorrow_price
            ),

            "future": future,

            "company": company,

            "rmse": float(rmse),

            "mae": float(mae),

            "mape": float(mape),

            "accuracy": float(accuracy),

            "predictions": predictions,

            "actual": actual

        }

    except Exception as e:

        print(e)

        return None