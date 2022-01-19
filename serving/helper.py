import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from transformer_layers import *


def load_lob_data(day=1):
    """
        Load the Limit Order Book (lob) data for the specified day.

        :param day: an integer between 1 and 3
        :return: returns a dictionary conataining 5 arrays (one for each asset) with the lob data for the day
    """

    current_path = os.getcwd()

    data_path = os.path.join(current_path, f"serving/static/data/lobdata_day_{day}.txt")


    lob_data = np.loadtxt(data_path)
    lob_data = np.array(lob_data[:40, :].T)

    # For all assets the data is in one array. Since no information about the position of each assets data is provided
    # splits are specified by finding large jumps in the best ask level.
    best_asks = lob_data[:, 0]
    difference_best_asks = best_asks - np.roll(best_asks, 1)
    split = (abs(difference_best_asks) > 0.03).nonzero()[0]
    split = np.append(split, len(best_asks))

    lob_data_assets = {"all": lob_data}
    for i in range(5):
        lob_data_assets[f"asset {i + 1}"] = lob_data[split[i]:split[i + 1], :]

    return lob_data_assets


def predict(lob_data, T=100, horizon=4):
    """
        Using the provided lob data predictions (up/down/stationary) are made about the direction of the assets price
        in the prediction horizon.

        :param lob_data: lob data as an array
        :param T: timesteps in the past used for prediction
        :param horizon: how far into the future the predictions should be made
        :return: returns a 1-d array with the predictions of the model (0: up, 1: stationary, 2: down)
    """

    current_path = os.getcwd()

    model_path = os.path.join(current_path, "serving/static/deeplob_serving_model.h5")

    model = load_model(model_path,
                       custom_objects={"PositionalEncodingLayer": PositionalEncodingLayer,
                                       "MultiHeadSelfAttention": MultiHeadSelfAttention,
                                       "TransformerTransition": TransformerTransition})
    # The lob data has to be batched together to resemble the input shape of the tensorflow model
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
    """
        The lob data and corresponding predictions are used to simulate an agent trading based on the predictions:
        following rules. At each time-step, our model generates a signal from the network outputs (−1, 0, +1) to
        indicate the price movements in k steps. Signals (−1, 0, +1) correspond to actions (sell, wait and buy). Suppose
        our model produces a prediction of +1 at time t, we then buy µ shares at time t+5 (taking slippage into account)
        and hold until −1 appears to sell all µ shares (we do nothing if 0 appears). We apply the same rule to short
        selling and repeat the process during a day. All positions are closed by the end of the day, so we hold no
        stocks overnight. We make sure no trades take place at the time of auction, so no abnormal profits are generated

        :param lob_data: lob data as an array
        :param predictions: 1-d array with the predictions of the model
        :param asset: which asset should be traded
        :param T: timesteps in the past used for prediction
        :return: a dataframe containing the simulated returns of the strategy and a dictionary containing portfolio KPIs
    """

    backtest_df = pd.DataFrame(data=lob_data[asset][:, [0, 2]], columns=["ask", "bid"])
    backtest_df = backtest_df.iloc[T - 1:, :]
    backtest_df["prediction"] = predictions

    backtest_df["shares_held"] = np.nan
    backtest_df.loc[backtest_df["prediction"] == 0, "shares_held"] = 1
    backtest_df.loc[backtest_df["prediction"] == 2, "shares_held"] = -1
    backtest_df["shares_held"] = backtest_df["shares_held"].shift(5, fill_value=0)
    backtest_df["shares_held"].fillna(method="ffill", inplace=True)

    backtest_df["ask_change"] = backtest_df["ask"] - backtest_df["ask"].shift(1)
    backtest_df["bid_change"] = backtest_df["bid"] - backtest_df["bid"].shift(1)

    backtest_df["return"] = 0
    backtest_df.loc[backtest_df["shares_held"] > 0, "return"] = backtest_df["bid_change"].copy()
    backtest_df.loc[backtest_df["shares_held"] < 0, "return"] = - backtest_df["ask_change"].copy()

    backtest_df["profit"] = backtest_df["return"].copy()
    backtest_df["profit"].iloc[0] = backtest_df["ask"].iloc[0].copy()
    backtest_df["cumulative_profit"] = backtest_df["profit"].cumsum()

    backtest_df["relative_return"] = (backtest_df["cumulative_profit"] - backtest_df["cumulative_profit"].shift(1)) / backtest_df["cumulative_profit"].shift(1)
    backtest_df["relative_return"].iloc[0] = 0
    backtest_df["cumulative_relative_return"] = (1 + backtest_df["relative_return"]).cumprod()

    backtest_df["cumulative_relative_return_long"] = backtest_df["cumulative_relative_return"].copy()
    backtest_df["cumulative_relative_return_short"] = backtest_df["cumulative_relative_return"].copy()

    backtest_df["cumulative_relative_return_long"].loc[backtest_df["shares_held"] < 0] = np.nan
    backtest_df["cumulative_relative_return_short"].loc[backtest_df["shares_held"] >= 0] = np.nan

    backtest_df["HODL"] = (1 + (backtest_df["ask_change"] / backtest_df["ask"].shift(1))).cumprod()

    backtest_df["shares_held_change"] = backtest_df["shares_held"].shift(-1, fill_value=0) - backtest_df["shares_held"]
    number_trades = backtest_df["shares_held_change"].abs().sum() / 2 + 1

    backtest_kpis = {"return": backtest_df["cumulative_relative_return"].iloc[-2],
                     "number_trades": number_trades}

    return backtest_df, backtest_kpis
