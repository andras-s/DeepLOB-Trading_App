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
    trading_day = st.slider("Select Trading Day", 1, 3)
    asset = st.selectbox(
        'Which asset do you want to trade?',
        ('asset 1', 'asset 2', 'asset 3', 'asset 4', 'asset 5'))

    plot_data = pd.Series(0)

    cumulative_return = ""
    alpha = ""
    number_trades = ""

    if st.button("Simulate Trading"):
        T = 100
        horizon = 4
        lob_data = load_lob_data(day=trading_day)
        predictions = predict(lob_data[asset], T=T, horizon=horizon)
        backtest_df, backtest_kpis = backtest(lob_data, predictions, asset=asset, T=100)

        plot_data = backtest_df["cumulative_profit"]

    fig, ax = plt.subplots()
    ax = plot_data.plot()
    ax.set(xlabel='Time [pip]', ylabel='Cumulative Return []')
    st.pyplot(fig)

    st.write(
        f"""
        ##
        Cumulative Return: {cumulative_return} \n
        Alpha: {alpha} \n
        Number of Trades: {number_trades}
        """)


if __name__ == '__main__':
    main()
