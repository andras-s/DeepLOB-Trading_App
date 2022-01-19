import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from helper import *


def main():

    st.title("Trading Bot")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Deep Learning High Frequency Trader</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write(
        """
        ##
        A deep learning model was trained to predict the the price movement (up/down/no movement) of 5 assets from 
        limit order book data. Based on these predictions a simple trading strategy is derived to trade on unseen days.
        
        The user can choose a day and an asset to see how the trading strategy would have performed.
        """)

    trading_day = st.slider("Select Trading Day", 1, 3)
    asset = st.selectbox(
        'Which asset do you want to trade?',
        ('asset 1', 'asset 2', 'asset 3', 'asset 4', 'asset 5'))

    plot_data = pd.DataFrame(columns=["cumulative_relative_return_long",
                                      "cumulative_relative_return_short",
                                      "HODL"])

    cumulative_return = ""
    number_trades = ""
    kpis = pd.DataFrame([cumulative_return, number_trades], columns=["value"], index=['Cumulative Return', 'Number of Trades'])

    # Upon pressing the button the lob data and model is loaded. Predictions are made and trading based on the
    # predictions is performed
    if st.button("Simulate Trading"):
        T = 100
        horizon = 4
        lob_data = load_lob_data(day=trading_day)
        predictions = predict(lob_data[asset], T=T, horizon=horizon)
        backtest_df, backtest_kpis = backtest(lob_data, predictions, asset=asset, T=100)

        plot_data = backtest_df[["cumulative_relative_return_long",
                                 "cumulative_relative_return_short",
                                 "HODL"]]
        cumulative_return = f"{round(backtest_kpis['return'], 3)}"
        number_trades = f"{round(backtest_kpis['number_trades'])}"
        kpis = pd.DataFrame([cumulative_return, number_trades], columns=["value"],
                            index=['Cumulative Return', 'Number of Trades'])

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(plot_data["cumulative_relative_return_long"], label='Bot Strategy Long', c="g", linewidth=1)
    ax.plot(plot_data["cumulative_relative_return_short"], label='Bot Strategy Short', c="r", linewidth=1)
    ax.plot(plot_data["HODL"], label='HODL Strategy', c="black", linewidth=1)
    ax.set(xlabel='Time [pip]', ylabel='Relative Return')
    ax.legend()
    ax.set_title("Strategy Performance")
    st.pyplot(fig)

    # Table with KPIs
    st.table(kpis)

    st.write(
        """
        Note: Trading costs and bid-ask spreads are not taken into account. The motivation for this app is simply
        to validate the performance of the prediction model.
        """)


if __name__ == '__main__':
    main()
