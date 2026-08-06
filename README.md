# 📈 AI Stock Price Prediction using LSTM

## Overview

This project predicts stock closing prices using a Long Short-Term Memory (LSTM) Deep Learning model built with TensorFlow and Keras.

The application downloads historical stock market data from Yahoo Finance, preprocesses it, trains an LSTM neural network, and predicts future stock prices. A Streamlit dashboard provides interactive visualizations and evaluation metrics.

---

## Features

- 📊 Historical stock data from Yahoo Finance
- 🤖 LSTM-based stock price prediction
- 📈 Interactive Plotly charts
- 🕯️ Candlestick chart
- 📉 50-day & 200-day Moving Averages
- 📅 Next-day stock price prediction
- 📊 Model evaluation using RMSE, MAE, and Accuracy
- 💻 Streamlit web dashboard
- 📥 Download processed stock data

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Joblib

---

## Project Structure

```
Stock-Prediction
│
├── app.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
```

---

## Installation

```bash
git clone https://github.com/SaiChandanabairoju/AI-Stock-Prediction.git

cd AI-Stock-Prediction

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python train.py

streamlit run app.py
```

---

## Future Improvements

- Multi-stock training
- 7-day forecasting
- Technical indicators (RSI, MACD)
- Transformer-based forecasting
- Real-time prediction

---

## Author

**Sai Chandana**