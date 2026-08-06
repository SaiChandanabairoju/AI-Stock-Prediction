import pandas as pd

def calculate_indicators(df):

    df = df.copy()

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA100"] = df["Close"].rolling(100).mean()

    df["MA200"] = df["Close"].rolling(200).mean()

    return df