import pandas as pd
import numpy as np
import os
import traceback
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from config import RUN as run_conf
import compute_indicators_labels_lib
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel


def calc_cum_ret_s1(x, stop_loss, fee):
    """
    compute cumulative return strategy s1.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss: Stop loss value
    :param fee: Transaction fee
    :return: history, capital, num_op, min_drawdown, max_gain
    """
    order_pending = 0
    price_start = 0
    capital = 1
    history = []
    min_drawdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0

    for row in x.itertuples():
        # Record current capital
        history.append(capital)

        # Handle stop loss if an order is pending
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                price_end = price_start * (1 - stop_loss)
                order_pending = 0
                pct_chg = (price_end - price_start) / price_start
                min_drawdown = min(min_drawdown, pct_chg)
                capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
                price_start = 0

        # Handle BUY label
        if row.label == BUY and not order_pending:
            order_pending = 1
            price_start = row.Close
            num_ops += 1
            continue

        # Handle SELL or HOLD label if order is pending
        if row.label in [SELL, HOLD] and order_pending:
            price_end = row.Close
            pct_chg = (price_end - price_start) / price_start
            if pct_chg > 0:
                good_ops += 1
            min_drawdown = min(min_drawdown, pct_chg)
            max_gain = max(max_gain, pct_chg)
            order_pending = 0
            capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
            price_start = 0

    # Handle last candle if order is pending
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)
        min_drawdown = min(min_drawdown, pct_chg)
        max_gain = max(max_gain, pct_chg)
        capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))

    return history, capital, num_ops, min_drawdown, max_gain, good_ops


def calc_cum_ret_s2(x, stop_loss, fee):
    """
    compute cumulative return strategy s2.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss: Stop loss value
    :param fee: Transaction fee
    :return: history, capital, num_op, min_drawdown, max_gain
    """
    order_pending = 0
    price_start = 0
    capital = 1
    history = []
    min_drawdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    fw_pos = 0

    for row in x.itertuples():
        # Record current capital
        history.append(capital)

        # Handle stop loss if an order is pending
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                price_end = price_start * (1 - stop_loss)
                order_pending = 0
                pct_chg = (price_end - price_start) / price_start
                min_drawdown = min(min_drawdown, pct_chg)
                capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
                price_start = 0
                fw_pos = 0

        # Handle BUY label
        if row.label == BUY and not order_pending:
            order_pending = 1
            price_start = row.Close
            num_ops += 1
            fw_pos = 0
            continue

        # Handle SELL or HOLD label if order is pending
        if row.label in [SELL, HOLD] and order_pending:
            price_end = row.Close
            pct_chg = (price_end - price_start) / price_start
            if pct_chg > 0:
                good_ops += 1
            min_drawdown = min(min_drawdown, pct_chg)
            max_gain = max(max_gain, pct_chg)
            order_pending = 0
            capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
            price_start = 0
            fw_pos = 0

    # Handle last candle if order is pending
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)
        min_drawdown = min(min_drawdown, pct_chg)
        max_gain = max(max_gain, pct_chg)
        capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))

    return history, capital, num_ops, min_drawdown, max_gain, good_ops


def calc_cum_ret_s3(x, stop_loss, fee, f_win):
    """
    compute cumulative return strategy s3.
    :param x: ordered prices timeseries of a coin and labels
    :param stop_loss: Stop loss value
    :param fee: Transaction fee
    :param f_win: Forward window size
    :return: history, capital, num_op, min_drawdown, max_gain
    """
    order_pending = 0
    price_start = 0
    capital = 1
    history = []
    min_drawdown = 0
    max_gain = 0
    num_ops = 0
    good_ops = 0
    fw_pos = 0

    for row in x.itertuples():
        # Record current capital
        history.append(capital)

        # Handle stop loss if an order is pending
        if order_pending:
            price_end = row.Low
            pct_chg = (price_end - price_start) / price_start
            if pct_chg < -stop_loss:
                price_end = price_start * (1 - stop_loss)
                order_pending = 0
                pct_chg = (price_end - price_start) / price_start
                min_drawdown = min(min_drawdown, pct_chg)
                capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
                price_start = 0
                fw_pos = 0

        # Handle forward window
        if order_pending:
            if fw_pos < f_win:
                fw_pos += 1
                continue
            elif fw_pos == f_win:
                if row.label == BUY:
                    fw_pos = 0
                    continue

        # Handle BUY label
        if row.label == BUY and not order_pending:
            order_pending = 1
            price_start = row.Close
            num_ops += 1
            fw_pos = 0
            continue

        # Handle SELL or HOLD label if order is pending
        if row.label in [SELL, HOLD] and order_pending:
            price_end = row.Close
            pct_chg = (price_end - price_start) / price_start
            if pct_chg > 0:
                good_ops += 1
            min_drawdown = min(min_drawdown, pct_chg)
            max_gain = max(max_gain, pct_chg)
            order_pending = 0
            capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))
            price_start = 0
            fw_pos = 0

    # Handle last candle if order is pending
    if order_pending:
        price_end = row.Low
        pct_chg = (price_end - price_start) / price_start
        if pct_chg < -stop_loss:
            price_end = price_start * (1 - stop_loss)
        min_drawdown = min(min_drawdown, pct_chg)
        max_gain = max(max_gain, pct_chg)
        capital *= 1 + (((price_end * (1 - fee)) - (price_start * (1 + fee))) / (price_start * (1 + fee)))

    return history, capital, num_ops, min_drawdown, max_gain, good_ops


def backtest_single_coin(RUN, filename, mdl_name="model.h5", suffix=""):
    """
    Backtest a coin whose timeseries is contained in filename.
    :param suffix:
    :param mdl_name:
    :param RUN: Configuration dictionary
    :param filename: CSV file containing coin timeseries data
    :return: a dictionary with dummy (du) and neural net (nn) statistics of backtest
    """
    try:
        # Load and prepare data
        data = pd.read_csv(f"{RUN['folder']}{filename}")
        data['Date'] = pd.to_datetime(data['Date'])
        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
        data = data[(data['Date'] >= RUN['back_test_start']) & (data['Date'] <= RUN['back_test_end'])]
        if data.empty:
            raise ValueError("Void dataframe")

        data.set_index('Date', inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data_pred = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', "Asset_name"])
        if data_pred.empty:
            raise ValueError("Void dataframe")

        # Load and scale data
        nr = joblib.load(f'scaler_{RUN["b_window"]}_{RUN["f_window"]}.pkl')
        Xs = nr.transform(data_pred)

        # Load model and predict
        model = NNModel(Xs.shape[1], 3)
        model.load(mdl_name)
        labels = model.predict(Xs)
        data['label'] = labels

        # Calculate cumulative returns for neural network strategy
        hist_nn, cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn = calc_cum_ret_s1(
            data, RUN['stop_loss'], RUN['commission fee']
        )

        # Predict with dummy model and calculate cumulative returns
        labels = model.dummy_predict(Xs)
        data['label'] = labels
        hist_du, cap_du, num_op_du, min_drawdown_du, max_gain_du, g_ops_du = calc_cum_ret_s1(
            data, RUN['stop_loss'], RUN['commission fee']
        )

        # Plot results
        dates = list(data.index)
        plt.rcParams['font.size'] = 14
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        ax = axs[0]
        ax.set_facecolor('#eeeeee')
        ax.plot(dates, np.log(hist_nn), label='MLP', color='green')
        ax.plot(dates, np.log(hist_du), label='Dummy', color='red')
        ax.set(xlabel='', ylabel='Log(Return)',
               title=filename.split('.')[0] + f" backW={RUN['b_window']}, forW={RUN['f_window']}")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.grid()
        ax.legend()

        ax = axs[1]
        ax.set_facecolor('#eeeeee')
        ax.plot(dates, data['Close'])
        ax.set(xlabel='', ylabel='Price', title="")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.grid()

        fig.savefig(RUN["reports"] + filename.split('.')[0] + f"_b{RUN['b_window']}_f{RUN['f_window']}_{suffix}.png")
        plt.show()

        return {'du': (cap_du, num_op_du, min_drawdown_du, max_gain_du, g_ops_du),
                'nn': (cap_nn, num_op_nn, min_drawdown_nn, max_gain_nn, g_ops_nn)}

    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)


if __name__ == "__main__":
    backtest_single_coin(run_conf, 'BTCUSDT.csv')
