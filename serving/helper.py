import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from transformer_layers import *


def load_lob_data(day=1):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, f"serving\static\data\lobdata_day_{day}.txt")

    lob_data = np.loadtxt(data_path)
    lob_data = np.array(lob_data[:40, :].T)

    best_asks = lob_data[:, 0]
    difference_best_asks = best_asks - np.roll(best_asks, 1)
    split = (abs(difference_best_asks) > 0.03).nonzero()[0]
    split = np.append(split, len(best_asks))

    lob_data_assets = {"all": lob_data}
    for i in range(5):
        lob_data_assets[f"asset {i + 1}"] = lob_data[split[i]:split[i + 1], :]

    return lob_data_assets


def predict(lob_data, T=100, horizon=4):
    current_path = os.getcwd()
    model_path = os.path.join(current_path, "serving\static\deeplob_serving_model.h5")
    model = load_model(model_path,
                       custom_objects={"PositionalEncodingLayer": PositionalEncodingLayer,
                                       "MultiHeadSelfAttention": MultiHeadSelfAttention,
                                       "TransformerTransition": TransformerTransition})
    [N, D] = lob_data.shape
    df = np.array(lob_data)
    input_data = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        input_data[i - T] = df[i - T:i, :]
    input_data = input_data.reshape(input_data.shape + (1,))

    predictions = model.predict(input_data)
    predicted_directions = np.argmax(predictions[:, horizon], axis=1)

    return predicted_directions


def backtest(lob_data, predictions, asset="asset 5", T=100):
    df = pd.DataFrame(data=lob_data[asset][:, [0, 2]], columns=["ask", "bid"])
    df = df.iloc[T - 1:, :]
    df["prediction"] = predictions

    df["shares_held"] = np.nan
    df.loc[df["prediction"] == 1, "shares_held"] = 1
    df.loc[df["prediction"] == 2, "shares_held"] = -1
    df["shares_held"] = df["shares_held"].shift(5, fill_value=0)
    df["shares_held"].fillna(method="ffill", inplace=True)

    df["ask_change"] = df["ask"] - df["ask"].shift(1)
    df["bid_change"] = df["bid"] - df["bid"].shift(1)

    df["return"] = 0
    df.loc[df["shares_held"] > 0, "return"] = df["bid_change"]
    df.loc[df["shares_held"] < 0, "return"] = df["ask_change"]

    df["shares_held_change"] = df["shares_held"].shift(-1, fill_value=0) - df["shares_held"]
    # df["cost"] = df["shares_held_change"] * (df["ask"] - df["bid"]) * (df["shares_held_change"] > 0)

    df["profit"] = df["return"]  # - df["cost"]
    df["cumulative_profit"] = df["profit"].cumsum()

    return df, 1
