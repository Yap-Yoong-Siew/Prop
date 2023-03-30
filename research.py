# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:40:54 2023

@author: user
"""

import sys
import re
import os
import glob
import winsound
from tabulate import tabulate
import PySimpleGUI as sg
from hyperopt import fmin, tpe, space_eval, Trials
from hyperopt import hp, STATUS_OK, STATUS_FAIL
import warnings
import vectorbtpro as vbt
import numpy as np
import pandas as pd
from numba import njit
from matplotlib import pyplot as plt
import datetime as dt
import time
from tqdm import tqdm, trange
import talib
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime, timedelta
import plotly.io as pio
import copy
import math
import pickle
import seaborn as sns
pio.renderers.default = 'browser'
warnings.simplefilter('ignore')

space = " "

# %% Functions


@njit
def signal_func_nb(c, entries, exits, short_entries, short_exits, max_dca=3):
    '''
    This function is for DCA, called in the from_signal
    Parameters
    ----------
    c : TYPE
        c is like a place holder for the dataframe, c[i] is the ith row
    entries : TYPE
        DESCRIPTION.
    exits : TYPE
        DESCRIPTION.
    short_entries : TYPE
        DESCRIPTION.
    short_exits : TYPE
        DESCRIPTION.
    max_dca : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    long_entry : TYPE
        DESCRIPTION.
    long_exit : TYPE
        DESCRIPTION.
    short_entry : TYPE
        DESCRIPTION.
    short_exit : TYPE
        DESCRIPTION.

    '''
    last_position = c.last_position
    last_order_price = c.order_records[c.order_counts[c.col] - 1, c.col]['price']

    if (entries[c.i][c.col]):  # long signal
        if(last_position[c.col] > 0):  # long signal with open long
            if(c.open[c.i][c.col] > last_order_price):
                entries[c.i][c.col] = False
            # long signal more than 3 # Constraint directly <from_signal>
            if(last_position[c.col] > max_dca):
                entries[c.i][c.col] = False

    if (short_entries[c.i][c.col]):  # short signal
        if(last_position[c.col] < 0):  # short signal with open short
            if(c.open[c.i][c.col] < last_order_price):
                short_entries[c.i][c.col] = False
            if(last_position[c.col] < -max_dca):  # long signal more than 3
                short_entries[c.i][c.col] = False

    long_entry = entries[c.i][c.col]
    long_exit = exits[c.i][c.col]
    short_entry = short_entries[c.i][c.col]
    short_exit = short_exits[c.i][c.col]

    return long_entry, long_exit, short_entry, short_exit

# Prepare folder to plot and store the graph


def prep_folder(folder="./vbt_pro/", delete=True):
    import os
    import shutil

    # Check if folder created
    if(os.path.exists(folder) == False):
        try:
            os.mkdir(folder)
        except OSError as error:
            print(error)
    return
    # Check if there are previous files need to delete
    if(delete == True):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print('Files are deleted')
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def plot_bt_rsi(pf, df_all_tf, rsi_fast, rsi_slow, entry_signal, exit_signal,
                short_entry_signal, short_exit_signal,
                entry_sd, exit_sd, Acc_symbol, show_fig=True, save_fig=False, text="BT"):
    fig = pf.plot(subplots=[
        # 'orders',
        'trades',
        'trade_pnl',
        ('Entry', dict(
            title='Entry Signal',
            yaxis_kwargs=dict(title='Entry'),
            check_is_not_grouped=True,
        )),
        ('cum_returns', dict(
            title='Cummulative Returns',
            yaxis_kwargs=dict(title='Cummulative Returns'),
            plot_func='plot_cum_returns',
            pass_hline_shape_kwargs=True,
            pass_add_trace_kwargs=True,
            pass_xref=True,
            pass_yref=True,
            bm_returns=False,
            check_is_not_grouped=True,
        )),
        ('Exit', dict(
            title='Exit Signal',
            yaxis_kwargs=dict(title='Exit'),
            check_is_not_grouped=True,
        )),
        'underwater'
    ],
        make_subplots_kwargs=dict(rows=3, cols=2, shared_xaxes='all'),
        column=Acc_symbol
    )
    scatter_ema = vbt.Scatter(
        data=rsi_fast,
        x_labels=df_all_tf.index,
        trace_names=["trades"],
        trace_kwargs=dict(line=dict(color='red')),
        add_trace_kwargs=dict(row=2, col=1),
        fig=fig
    )

    scatter_slow = vbt.Scatter(
        data=rsi_slow,
        x_labels=df_all_tf.index,
        trace_names=["trades"],
        trace_kwargs=dict(line=dict(color='blue')),
        add_trace_kwargs=dict(row=3, col=1),
        fig=fig
    )

    scatter_entry = vbt.Scatter(
        data=entry_signal,
        x_labels=df_all_tf.index,
        trace_names=["Long Entry"],
        trace_kwargs=dict(line=dict(color='green')),
        add_trace_kwargs=dict(row=2, col=1),
        fig=fig
    )

    scatter_exit = vbt.Scatter(
        data=exit_signal,
        x_labels=df_all_tf.index,
        trace_names=["Long Exit"],
        trace_kwargs=dict(line=dict(color='red')),
        add_trace_kwargs=dict(row=3, col=1),
        fig=fig
    )

    scatter_entry_short = vbt.Scatter(
        data=short_entry_signal,
        x_labels=df_all_tf.index,
        trace_names=["Short Entry"],
        trace_kwargs=dict(line=dict(color='red')),
        add_trace_kwargs=dict(row=2, col=1),
        fig=fig
    )

    scatter_exit_short = vbt.Scatter(
        data=short_exit_signal,
        x_labels=df_all_tf.index,
        trace_names=["Short Exit"],
        trace_kwargs=dict(line=dict(color='magenta')),
        add_trace_kwargs=dict(row=3, col=1),
        fig=fig
    )

    fig = fig.add_hline(
        y=entry_sd,
        line_color="black",
        row=[2, 3],
        col=[1, 1],
        line_width=1
    )

    fig = fig.add_hline(
        y=100-entry_sd,
        line_color="black",
        row=[2, 3],
        col=[1, 1],
        line_width=1
    )

    fig = fig.add_hline(
        y=exit_sd,
        line_color="blue",
        row=[2, 3],
        col=[1, 1],
        line_width=1
    )

    fig = fig.add_hline(
        y=100-exit_sd,
        line_color="blue",
        row=[2, 3],
        col=[1, 1],
        line_width=1
    )
    if(show_fig):
        fig.show()


@njit
def rsi_signal_nb_2d(rsi, rsi_thresh_entry, rsi_thresh_exit, rsi_candle,
                     rsi_cross_entry, rsi_cross_exit):
    '''
    helper function for rsi_custom_nb, which is used for trading logic
    this helper function is where most trading logic sits

    Parameters
    ----------
    rsi : TYPE
        DESCRIPTION.
    rsi_thresh_entry : TYPE
        DESCRIPTION.
    rsi_thresh_exit : TYPE
        DESCRIPTION.
    rsi_candle : TYPE
        DESCRIPTION.
    rsi_cross_entry : TYPE
        DESCRIPTION.
    rsi_cross_exit : TYPE
        DESCRIPTION.

    Returns
    -------
    rsi_signal_buy : TYPE
        DESCRIPTION.
    rsi_signal_sell : TYPE
        DESCRIPTION.
    rsi_signal_close_buy : TYPE
        DESCRIPTION.
    rsi_signal_close_sell : TYPE
        DESCRIPTION.

    '''

    # -- Pre-Process Variables
    rsi_shape = rsi.shape
    # numpy array with the same shape as rsi
    lower_entry_thresh = np.full(rsi_shape, rsi_thresh_entry)
    upper_exit_thresh = np.full(rsi_shape, rsi_thresh_exit)
    upper_entry_thresh = np.full(rsi_shape, 100-rsi_thresh_entry)
    lower_exit_thresh = np.full(rsi_shape, 100-rsi_thresh_exit)
    # print(rsi)
    # print(rsi.shape)
    # print(lower_exit_thresh, lower_exit_thresh.shape)

    # -- Entry Signal RSI Cross
    if(rsi_cross_entry == 1):  # "Cross Up Buy" # late entry
        rsi_signal_buy = vbt.nb.crossed_above_nb(rsi, lower_entry_thresh)
        rsi_signal_sell = vbt.nb.crossed_above_nb(upper_entry_thresh, rsi)
    elif(rsi_cross_entry == 2):  # "Cross Down Buy" # early entry
        rsi_signal_buy = vbt.nb.crossed_above_nb(lower_entry_thresh, rsi)
        rsi_signal_sell = vbt.nb.crossed_above_nb(rsi, upper_entry_thresh)
    elif(rsi_cross_entry == 3):  # "Cross Up Sell" highly unlikely
        rsi_signal_sell = vbt.nb.crossed_above_nb(rsi, lower_entry_thresh)
        rsi_signal_buy = vbt.nb.crossed_above_nb(upper_entry_thresh, rsi)
    elif(rsi_cross_entry == 4):  # "Cross Down Sell" highly unlikely
        rsi_signal_sell = vbt.nb.crossed_above_nb(lower_entry_thresh, rsi)
        rsi_signal_buy = vbt.nb.crossed_above_nb(rsi, upper_entry_thresh)
        #3 & 4 is the same as 1 & 2 but flip the signal from buy to sell

    rsi_signal_buy = vbt.nb.fshift_nb(rsi_signal_buy, rsi_candle + 1)
    rsi_signal_sell = vbt.nb.fshift_nb(rsi_signal_sell, rsi_candle + 1)

    # -- Exit Signal RSI Cross
    if(rsi_cross_entry <= 2):
        if(rsi_cross_exit == 1):  # "Cross Up Buy Close" This mean early close
            rsi_signal_close_buy = vbt.nb.crossed_above_nb(
                rsi, upper_exit_thresh)
            rsi_signal_close_sell = vbt.nb.crossed_above_nb(
                lower_exit_thresh, rsi)
        elif(rsi_cross_exit == 2):  # "Cross Down Buy Close" This means late close
            rsi_signal_close_buy = vbt.nb.crossed_above_nb(
                upper_exit_thresh, rsi)
            rsi_signal_close_sell = vbt.nb.crossed_above_nb(
                rsi, lower_exit_thresh)
    elif(rsi_cross_entry <= 4):  # If Entry Signal is Reversed These 2 are highly improbably
        if(rsi_cross_exit == 1):  # "Cross Up Buy Close"
            rsi_signal_close_sell = vbt.nb.crossed_above_nb(
                rsi, upper_exit_thresh)
            rsi_signal_close_buy = vbt.nb.crossed_above_nb(
                lower_exit_thresh, rsi)
        elif(rsi_cross_exit == 2):  # "Cross Down Buy Close"
            rsi_signal_close_sell = vbt.nb.crossed_above_nb(
                upper_exit_thresh, rsi)
            rsi_signal_close_buy = vbt.nb.crossed_above_nb(
                rsi, lower_exit_thresh)

    return rsi_signal_buy, rsi_signal_sell, rsi_signal_close_buy, rsi_signal_close_sell


def rsi_custom_nb(price,
                  rsi_fast_period=14,
                  rsi_slow_period=15,
                  rsi_thresh_entry=30.0,
                  rsi_thresh_exit=60.0,
                  rsi_candle=0,
                  rsi_cross_entry=1,
                  rsi_cross_exit=1,
                  ):
    '''
    This function is called in the vbt.IF, with_apply_func(here, xxx, .....
    The idea is the vbt.IF is the shell and this function contains 
    the trading logic. (with help from rsi_signal_nb_2d helper function)

    takes in price data, and the parameters that is tunable
    splits 
        DESCRIPTION. The default is 14.
    '''

    # This RSI method can process 2D array
    rsi_fast = vbt.talib('RSI').run(
        price, timeperiod=rsi_fast_period).real.to_numpy()
    rsi_slow = vbt.talib('RSI').run(
        price, timeperiod=rsi_slow_period).real.to_numpy()

    rsi = np.sum([rsi_fast, rsi_slow], axis=0) / 2
    # rsi = rsi_slow # For simplicity 1st

    # -- Trading Signal
    rsi_signal_buy, rsi_signal_sell, rsi_signal_close_buy, rsi_signal_close_sell = \
        rsi_signal_nb_2d(rsi, rsi_thresh_entry, rsi_thresh_exit, rsi_candle,
                         rsi_cross_entry, rsi_cross_exit)

    long_entries = rsi_signal_buy
    short_entries = rsi_signal_sell

    long_exit = rsi_signal_close_buy
    short_exit = rsi_signal_close_sell

    return long_entries, short_entries, long_exit, short_exit


# Indicator factory is the signal generator, at the end, the cross is the signal and is return
RSI_Custom = vbt.IF(  # the design is the vbt.IF is the shell, the first function following
    class_name='RSI_Custom',  # with_apply_func( is the trading logic)
    short_name='rsi_cus',
    input_names=['price'],
    param_names=['rsi_fast_period', 'rsi_slow_period', 'rsi_thresh_entry', 'rsi_thresh_exit',
                 'rsi_candle', 'rsi_cross_entry', 'rsi_cross_exit'],
    output_names=['long_entries', 'short_entries', 'long_exit', 'short_exit']
).with_apply_func(
    rsi_custom_nb,
    rsi_fast_period=14,
    rsi_slow_period=15,
    rsi_thresh_entry=30.0,
    rsi_thresh_exit=60.0,
    rsi_candle=0,
    rsi_cross_entry=1,
    rsi_cross_exit=1,
)

# Return backtest result


def bt_performance(pf):
    '''
    return backtest result, how good is a result of backtesting depends on linearity

    Parameters
    ----------
    pf : TYPE
        Portfolio object

    Returns
    -------
    profit_usd : TYPE
        profit in usd
    profit : TYPE
        profit in percentage
    max_drawdown : TYPE
        max drawdown in percentage
    profit_dd : TYPE
        profit to drawdown ratio
    win_rate : TYPE
        win rate in percentage
    Total_Closed_trades : TYPE
        number of closed trades
    r_sq : TYPE
        r_sq is the linearity in percentage

    '''
    init_cap = pf.init_cash.mean()
    # Net profit diff with TV
    profit = (pf.get_cumulative_returns().iloc[-1]-1).median()*100
    profit_usd = init_cap * profit / 100
    max_drawdown = (pf.get_drawdown().min()*-100).median()
    profit_dd = np.nan  # profit to drawdown ratio
    if(max_drawdown != 0):
        profit_dd = profit/max_drawdown  # Condition cater for multiple array

    trades_list = pf.trades.records_readable
    trade_pnl = trades_list['PnL']
    Total_Closed_trades = (
        pf.trades.records_readable['Status'] == 'Closed').sum()
    Total_trades = len(pf.trades.records_readable.index)

    win_rate = 0
    if(Total_trades != 0 and Total_Closed_trades != 0):
        if(Total_trades == Total_Closed_trades):  # Only calculate closed profit trade on WR
            profit_trade = (trade_pnl > 0).sum()
        else:  # If the last trade is not closed, exclude last trade
            trade_pnl = trade_pnl[:-1]
            profit_trade = (trade_pnl > 0).sum()
            profit_usd = trade_pnl.sum()
            if(init_cap != 0):
                profit = trade_pnl.sum() / init_cap * 100
        win_rate = profit_trade / Total_Closed_trades * 100  # Calculate WR in %

    # Linear Regression - Cost = 10ms
    r_sq = 0
    if(Total_Closed_trades > 1):
        pnl_cumsum = trade_pnl.cumsum()
        model = LinearRegression(fit_intercept=True)  # Fixed intercept at 0
        x = np.arange(len(trade_pnl)).reshape(-1, 1)
        x = x + 0.005*x*x  # Quadratic grow equity graph
        model.fit(x, pnl_cumsum)
        r_sq = model.score(x, pnl_cumsum) * 100 * np.sign(profit)

    return profit_usd, profit, max_drawdown, \
        profit_dd, win_rate, Total_Closed_trades, r_sq


def rsi_bt_ho_ohlc(price, setting):
    '''
    This function wrap all the backtesting code together
    This function == 1 epoch 
    When you call this code, you run 1 of many possible settings
    In sample training requires this function to be called with many possible settings



    Parameters
    ----------
    data_ohlc : TYPE
        DESCRIPTION.
    setting : TYPE
        this is a HyperOPT type

    Returns
    -------
    Pandas Series
        It will return that epoch's setting's result, kinda like loss values to choose from.

    '''
    # open_price = data_ohlc.get('Open')
    # close_price = data_ohlc.get('Close')
    # high_price = data_ohlc.get('High')
    # low_price = data_ohlc.get('Low')
    # price = open_price
    # print(price)
    
    
    #locking for market neural and computational efficiency
    # thresh need to sum to 100 to be market neural
    # rsi fast must be smaller than slow
    
    #fast = 2 - 35
    # gap, fast + gap = slow = 0 - 20
    
    # threshold = 15 - 40
    #market non neurality = -10 - 10
    #rsi_thresh_entry = threshold + market non neurality
    #rsi_thresh_exit = 100 - threshold
    open_price = price.get('Open').to_frame()
    close_price = price.get('Close').to_frame()
    high_price = price.get('High').to_frame()
    low_price = price.get('Low').to_frame()
    price = price.get('Open').to_frame()

    rsi_fast_period = int(setting.get('rsi_fast_period'))
    rsi_slow_period = int(setting.get('rsi_slow_period'))
    rsi_thresh_entry = setting.get('rsi_thresh_entry') 
    rsi_thresh_exit = setting.get('rsi_thresh_exit') 
    rsi_candle = int(setting.get('rsi_candle'))
    rsi_cross_entry = int(setting.get('rsi_cross_entry'))
    rsi_cross_exit = int(setting.get('rsi_cross_exit'))
    max_trade = int(setting.get('max_trade'))
    tp_percent = setting.get('tp_percent') / 10
    sl_percent = setting.get('sl_percent') / 10

    # tic = time.perf_counter()
    res_indi = RSI_Custom.run(price.to_numpy(),
                              rsi_fast_period=rsi_fast_period,
                              rsi_slow_period=rsi_slow_period,
                              rsi_thresh_entry=rsi_thresh_entry,
                              rsi_thresh_exit=rsi_thresh_exit,
                              rsi_candle=rsi_candle,
                              rsi_cross_entry=rsi_cross_entry,
                              rsi_cross_exit=rsi_cross_exit,
                              )
    # toc = time.perf_counter()
    # print(f"Build finished in {(toc - tic):0.4f} secs \n")

    long_entries = np.array(res_indi.long_entries, dtype=bool)
    short_entries = np.array(res_indi.short_entries, dtype=bool)

    long_exit = np.array(res_indi.long_exit, dtype=bool)
    short_exit = np.array(res_indi.short_exit, dtype=bool)

    pf = vbt.Portfolio.from_signals(
        close=close_price,
        open=open_price,
        high=high_price,
        low=low_price,
        price=open_price,
        signal_func_nb=signal_func_nb,
        signal_args=(long_entries, long_exit, short_entries,
                     short_exit, max_trade),
        size = 1,
        size_type='Amount',
        accumulate='Both',  # Allow Partial Close (Remove bit by bit)
        # accumulate='AddOnly', # Same with TV - Close all directly
        upon_opposite_entry='Reverse',
        upon_long_conflict='entry',
        upon_short_conflict='entry',
        init_cash=100_000,  # Fixed cash easier to compare
        sl_stop=sl_percent/100,
        tp_stop=tp_percent/100,
        fixed_fees=0.03,
        # max_size = 1, # Constraint max num of trade # Cannot apply if change of (lot) size / Diff Market
        # Automatic calculate frequency of dataset
        freq=price.index[1] - price.index[0]
    )

    profit_usd, profit, max_drawdown, profit_dd, win_rate, \
        Total_Closed_trades, r_sq = bt_performance(pf)
    expectancy = pf.trades.expectancy[0]
    # - Returning huge data will slow down the loop by 2x (Processing cost very minimal)
    return pd.Series([profit_usd, profit, max_drawdown, profit_dd, win_rate,
                      Total_Closed_trades, r_sq, expectancy, pf],
                     index=['Profit USD', 'Profit %', 'Max DD', 'Profit/DD', 'Win Rate',
                            'Total Closed Trades', 'Linear', 'expectancy', 'PF'])

# -- HyperOpt Objective Function

def objective(price_data):
    def objective_inner(args):

        res_bt = rsi_bt_ho_ohlc(price_data, args)
        freq = price_data.index[1] - price_data.index[0]
        freq = int(freq.seconds / 60)
        period = price_data.index[-1] - price_data.index[0]
        days = int(period.days)
        months = int(days / 30)
        # A formula to determine if the result is good based on the total trades and freq
        # if freq is 5 minute, then the total trades should be at least 60  per month
        # if freq is 15 minute, then the total trades should be at least 20  per month
        # if freq is 60 minute, then the total trades should be at least 5  per month
        # formula = 60 * 5 / freq
        
        pf = res_bt['PF']
        ## Period
        # period = pf.returns.index[-1] - pf.returns.index[0] # timedelta format
        # timedelta to number of days
        # period = period.days
        
        Total_Closed_trades = (pf.trades.records_readable['Status'] == 'Closed').sum().min() # Minimum trade count on all col
        
        monthly_trade = Total_Closed_trades / months
        
        min_trades_required = 60 * 5 / freq

        Trades_sqrt = 0
        if(Total_Closed_trades>0): Trades_sqrt = np.sqrt(res_bt['Total Closed Trades'].mean())
        profit_factor = pf.stats(metrics='profit_factor').mean()
        losing_streak = max(1,pf.trades.get_losing_streak().max().mean()) # https://vectorbt.pro/pvt_dc4c9ec9/api/portfolio/trades/#vectorbtpro.portfolio.trades.Trades.losing_streak
        dd = max(1e-8, res_bt['Max DD'].mean()) # Prevent 0 divide
        linear = abs(res_bt['Linear']) / 100 # Prevent overly large number
        expectancy = pf.trades.expectancy[0] 
        win_rate = res_bt['Win Rate'] / 100
        sharpe_ratio = pf.get_sharpe_ratio()
        # print out the expectancy, linear, trades_sqrt, profit_factor, win_rate, dd, losing_streak
        # print(f"expectancy: {expectancy}, linear: {linear}, Trades_sqrt: {Trades_sqrt}, profit_factor: {profit_factor}, win_rate: {win_rate}, dd: {dd}, losing_streak: {losing_streak}")
        # result_val = -1*res_bt['Profit %']*(abs(res_bt['Linear']))
        # result_val = -1*pf.trades.expectancy*(abs(res_bt['Linear']))
        result_val = -1*( (expectancy*linear*profit_factor*sharpe_ratio) / (dd*losing_streak) ) # Complex criteria that include all factors
        result_val = result_val.mean() # Mean performance on all col
        result_val = np.cbrt(monthly_trade / min_trades_required) * result_val # Scale up the result based on the trade surplus, cbrt is cube root
        if(result_val==-np.inf): result_val = np.inf # Too less trade cause wrong calculation
        
        if(monthly_trade < min_trades_required):
            result_val = 0
            return {'loss': result_val, 'status': STATUS_FAIL, 'res_bt': res_bt, 'Setting': args}
            # return 999 # Reverse Error | False Profit Signal
        return {'loss': result_val, 'status': STATUS_OK, 'res_bt': res_bt, 'Setting': args}
    return objective_inner

def time_rounder(t, Acc_tf):
    time_delta = int(re.findall(r'\d+', Acc_tf)[0])
    time_unit = ''.join(
        [i for i in Acc_tf if not i.isdigit()]).replace(" ", "")

    if(time_unit == "min"):  # Round to nearest next delta (in their unit)
        return (t.replace(second=0, microsecond=0, minute=t.minute - t.minute % (time_delta)) +
                pd.Timedelta(time_delta, unit=time_unit))
    elif(time_unit == "hour"):  # Round to nearest next delta (in their unit)
        return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour - t.hour % (time_delta)) +
                pd.Timedelta(time_delta, unit=time_unit))
    elif(time_unit == "D"):  # Round to nearest next delta (in their unit)
        return (t.replace(second=0, microsecond=0, minute=0, hour=0, day=t.day - t.day % (time_delta)) +
                pd.Timedelta(time_delta, unit=time_unit))



def update_row_with_dict(df, dictionary, idx):
    df.loc[idx, dictionary.keys()] = dictionary.values()

def read_csv_to_df(file_path):
    # Read the csv file into a pandas dataframe
    df = pd.read_csv(file_path, header=None, names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"], index_col=False)
    
    # Concatenate the "Date" and "Time" columns to create a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    
    # Set the "Datetime" column as the index of the dataframe
    df.set_index('datetime', inplace=True)
    
    # Drop the Date & Time column
    df = df.drop(['Date', 'Time'], axis=1)

    return df

def get_data(Acc_symbol, Acc_tf, range_, limit=25000):
    data = None
    # Loop until retrived data from TV (Might disconnect if running parallelly)
    while data is None:
        try:
            data = vbt.TVData.fetch(
                [Acc_symbol],
                timeframe=Acc_tf,
                limit=limit,
                show_progress=False)  # Clean process

        except Exception as e:
            print("Error in getting TV Data. Reason =", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            pass
    
    data = data.get('Open')
    splitter = splitter_wrapper(data, 400, 200)
    a = splitter.take(data)
    return data


def splitter_wrapper(price, is_bar_length, oos_bar_length): #is = In sample, oos = Out of sample
    return vbt.Splitter.from_rolling(
        price.index,
        length= is_bar_length + oos_bar_length,
        split= is_bar_length / (is_bar_length + oos_bar_length),
        offset=oos_bar_length - is_bar_length,
        set_labels=["IS", "OOS"]
    )

# generate a variable with a timpstamp of 2014-01-01 00:00:00



def range_backtest(is_price, oos_price, bt_result): # 1 range = 1 IS or 1 OOS, is either one, not both together

    pass
    
def splitter_backtest(is_slices, oos_slices): # splitter object, and  the open price
    # is_splitter = splitter['IS']    
    # oos_splitter = splitter['OOS']
    # is_slices = is_splitter.take(price) #insert price data inside the object and do the splitting
    # oos_slices = oos_splitter.take(price)
    i = 0 #take the last cutI 
    bt_result = pd.DataFrame()
    # for i in range(len(is_slices)):
    # range_backtest(is_slices[i][0].to_frame(), oos_slices[i][0].to_frame(), bt_result)
    range_backtest(is_slices.to_frame(), oos_slices.to_frame(), bt_result)

# write a function that trim the data to the same start and end date for all dataframes, the function will take in multiple dataframes
def trim_data(raw_dict, start=None, end=None, period=None):
    _raw_dict = copy.deepcopy(raw_dict)
    if start and end and period==None:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        for symbol, price_df in _raw_dict.items():
            _raw_dict[symbol] = price_df.loc[start_date:end_date]
            
    elif start and period and end==None:
        start_date = pd.to_datetime(start)
        end_date = start_date + pd.DateOffset(months=period)
        
        for symbol, price_df in _raw_dict.items():
            _raw_dict[symbol] = price_df.loc[start_date:end_date]
            
    elif start==None and period==None and end==None:
        pass
    else:
        # assert that either start and end or start and period are provided
        raise ValueError("Either start and end or start and period must be provided")
        
# new_dict = {symbol: price_df for symbol, price_df in _raw_dict.items()}
    start_date = max([price_df.index.min() for symbol, price_df in _raw_dict.items()])
    end_date = min([price_df.index.max() for symbol, price_df in _raw_dict.items()])
    print(type(start_date))
    print(end_date)
    # modify the dataframes in place instead of returning
    for symbol, price_df in _raw_dict.items():
        local_df = price_df.loc[start_date:end_date]
        # set the freq
        local_df = local_df.asfreq(price_df.index[1] - price_df.index[0])
        # forward fill and backward fill
        local_df = local_df.fillna(method='ffill').fillna(method='bfill')
        _raw_dict[symbol] = local_df
    # convert args to dictionary format


    # wrap dict to a vectorbt data fromat using from_data
    return vbt.Data.from_data(_raw_dict) # vbt_data




# write a function call hyperopt_factory, that takes in a dictionary of prices and returns a dictionary of top 100 best settings for each ticker
def hyperopt_factory(vbt_data, max_evals=5000, top_N=100):
    # list of symbols in vbt_data
    symbol_list = vbt_data.symbols
    # create a dictionary to store the results
    best_params_dict = {}
    # data is a dictionary of prices, keys are the tickers, values are the prices, the prices are in vectorbt format, with the index as datetime, and the columns as the tickers, do a for loop to iterate through the dictionary
    for symbol in symbol_list:
        
        trials = Trials()
        objective_closure = objective(vbt_data.data[symbol])
        best_params = fmin(objective_closure, settings_param, algo=tpe.suggest,
                            max_evals=max_evals, trials=trials)

        space_eval(settings_param, best_params)

        bt_result_trial = trials.results
        
        loss = []
        for trials in bt_result_trial:
            if (trials.get('status') == 'fail'):
                loss.append(20)
            else:
                loss.append(round(trials.get('loss'), 4))
        loss_series = pd.Series(loss)
        loss_series_filtered = loss_series.drop_duplicates()
        loss_series_filtered = loss_series_filtered.nsmallest(top_N)
        
        best_params_list = []
        for i, index in enumerate(loss_series_filtered.index):
        
            best_params_list.append(bt_result_trial[index].get("Setting"))
            
        # convert best_params_list to a series
        best_params_series = pd.Series(best_params_list) # series of series
        # save the best_params_series to the dictionary
        best_params_dict[symbol] = best_params_series
    return best_params_dict
        
    
# write a function that will take in 5 variables ticker, open, high, low, close and return a dataframe with the OHLC data of that ticker
def select_ticker(ticker, open, high, low, close): 
    open = open.loc[:, ticker]
    high = high.loc[:, ticker]
    low = low.loc[:, ticker]
    close = close.loc[:, ticker]
    # return a dataframe with the OHLC data of that ticker
    return pd.concat([open, high, low, close], axis=1, keys=['Open', 'High', 'Low', 'Close'])
    
# a function that takes in a best_settings_dict, vbt_price and return the pnl (cumsum) of the all the best settings for each ticker (dict)
def best_settings_dict_to_pnl(best_settings_dict, vbt_data):
    
    symbol_list = vbt_data.symbols
    
    # create a dictionary to store the results which is the pnl of each ticker's top 100 best settings, pnl calculated using cumsum
    pnl_dict = {}
    
    for symbol in symbol_list:
        price_data = vbt_data.data[symbol]
        # loop through the best settings for each ticker
        
        # create a list to store the pnl of each setting
        pnl_list = []
        
        for setting in best_settings_dict[symbol]:
            res_bt = rsi_bt_ho_ohlc(price_data, setting)
            
            pf = res_bt['PF']

            trades = pf.trades.records_readable
            trades_plot = trades.set_index('Exit Index')
            cumsum_series = trades_plot.groupby('Column')['PnL'].cumsum()
            # make sure the series index is unique and got no duplicate
            cumsum_series = cumsum_series.groupby(cumsum_series.index).first()
            # save the best setting's pnl to the dictionary
            pnl_list.append(cumsum_series)
            
        # convert the list to a series
        pnl_series = pd.Series(pnl_list)
        # save the series to the dictionary
        pnl_dict[symbol] = pnl_series
    return pnl_dict # dictionary of series of series, 
#keep in mind that pnl_dict contains pnl that is raw and there will be gap for different tf results 
# which is needed to be processed in the next step

# function that takes in a pnl_dict and plot the pnl of each ticker's top 100 best settings on the same plot
def monte_carlo(pnl_dict):
    # loop through the pnl_dict create a figure and plot the pnl of each ticker's top 100 best settings on the same plot
    for symbol, pnl_series in pnl_dict.items():
        fig = plt.figure()
        plt.title(symbol)
        for i, pnl in enumerate(pnl_series):
            if (i < 5):
                # get the sharpe ratio of the PnL
                sharpe = pnl.pct_change()
                sharpe = sharpe.mean() / sharpe.std() * np.sqrt(252*60*12)
                # set the sharpes as the label
                plt.plot(pnl, label=sharpe)
            else:
                plt.plot(pnl)
        plt.show()
    

#function to calculate correlation matrix when given pnl dict
def pnl_to_corr_matrix(pnl_dict, plot=False):
    

    pnl_dict_df = pd.DataFrame(pnl_dict)

    list_of_df = []
    
    for symbol in pnl_dict.keys():
        temp_df = pnl_dict_df.loc[:, symbol].apply(pd.Series).T
        temp_df = temp_df.rename(columns={col: symbol + '_' + str(col) for col in temp_df.columns})
        list_of_df.append(temp_df)
    portfolio_df = pd.concat(list_of_df, axis=1)
    
    # forward fill 
    portfolio_df = portfolio_df.fillna(method='ffill')
    # backward fill with 0
    portfolio_df = portfolio_df.fillna(0)
    
    
    # calculate the percentage change
    pct_change = portfolio_df.pct_change()
    # drop the first row of data
    pct_change = pct_change.drop(pct_change.index[0])
    # calculate the correlation matrix
    corr_matrix = pct_change.corr()
    
    if(plot):
        # plot the correlation matrix
        
        plt.figure(figsize=(30,30))
        
        sns.heatmap(corr_matrix, annot=False)
        plt.show()
    return corr_matrix

# write a function that takes in a correlation matrix and grid size and return a list of ticker pairs that are minimally correlated

def corr_matrix_to_portfolio_symbols(corr_matrix, vbt_data, grid_size):
    # this matrix is a symmetric matrix, so we only need to take the upper triangle
    # corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # take the smallest pairs from each grid except the diagonal
    small_df_matrices = {(int(i/grid_size),int(j/grid_size)): corr_matrix.iloc[i:i+grid_size, j:j+grid_size] for i in range(0, len(corr_matrix), grid_size) for j in range(0, len(corr_matrix), grid_size)}
    
    # get the list of symbols
    symbol_list = vbt_data.symbols
    
    portfolio_pairs = []
    # enumerate the symbols using for loop
    for i, symbol in enumerate(symbol_list):
        for j, symbol in enumerate(symbol_list):
            if(i >= j):
                # skip the diagonal
                continue
            else:
                # get the coresponding matrix
                sub_matrix = small_df_matrices[(i,j)]
                # get the smallest pair
                smallest_pair = sub_matrix.unstack().sort_values().index[0]
                portfolio_pairs.append(smallest_pair)
    # portfolio_pairs is a list of tuples, convert it to just a list
    portfolio_symbols = [elem for tup in portfolio_pairs for elem in tup]
    
    return portfolio_symbols

# write a function that takes in a portfolio_symbols (list), best_params_dict Dict('symbol' : 'pandas Series of settings'), vbt_price, do the backtest using rsi_bt_ho_ohlc and return a plot of the pnl backtest for each element in portfolio_symbols list, the  portfolio_symbols (list) contain elements thathave names such as 'EURUSD_1', which is split into 'EURUSD' and '1' and used to get the best_params_dict for that symbol
def portfolio_backtest(portfolio_symbols, best_params_dict, vbt_data, to_csv = False, save_plot = False, title="Title"):
    # create a list to store the results
    results = []
    production_setting = pd.DataFrame()
    # loop through the portfolio symbols
    for symbol in portfolio_symbols:
        # split the symbol into symbol and tf
        ticker, tf, id = symbol.split('_')
        # get the best params for that symbol
        best_params_series = best_params_dict[ticker + '_' + tf]
        # get the best params for that id
        best_params = best_params_series[int(id)] # is a dict
        # create a pd.Series to store the ticker(str) and tf(str) and id(str) and best_params(pd.Series)
        production_setting_params = pd.Series([ticker, tf, id], index=['Symbol', 'Timeframe', 'id'])
        production_setting_params = production_setting_params.append(pd.Series(best_params))
        production_setting = production_setting.append(production_setting_params, ignore_index=True)
        # get the price data for that symbol
        price_data = vbt_data.data[ticker + '_' + tf]
        
        # do the backtest using the best_params
        res_bt = rsi_bt_ho_ohlc(price_data, best_params)
        
        # get the portfolio
        pf = res_bt['PF']
        # get the trades
        trades = pf.trades.records_readable
        # set the index to the exit index
        trades_plot = trades.set_index('Exit Index')
        # calculate the cumsum
        cumsum_series = trades_plot.groupby('Column')['PnL'].cumsum()
        # change the series name to the symbol
        cumsum_series.name = symbol + '_' + tf
        # save the results to the list
        results.append(cumsum_series)
    # return results
    # find the mean of the results series while ignoring the NaN values
    # check if the results is only 1 column edge case
    
    if len(results) > 1:
        results_df = pd.concat(results, axis=1)
    else:
        results_df = pd.DataFrame(results)
    
    results_df = results_df.fillna(method='ffill').fillna(0)

    
    fig, ax = plt.subplots(figsize=(20, 10))

    # loop through each column in the dataframe
    for i in range(len(results_df.columns)):
        # plot the column
        column_data = results_df.iloc[:, i]
        # check the average (mean) of the column_data, if it is less than 1000, then multiple the column_data by 10 before plotting
        # if(column_data.mean() < 150):
        #     column_data = column_data * 10
        #     results_df.iloc[:, i] = column_data
        column_data.plot(ax=ax, label=column_data.name)
    results_df_mean = results_df.mean(axis=1)    
        # set the legend
    results_df_mean.plot(ax=ax, label="Mean", color="black")
    ax.legend()
    ax.set_xlabel("Exit Index")
    ax.set_ylabel("PnL")
    # add title
    ax.set_title(title)
    
    if(to_csv):
        print(production_setting)
        # find the sl_percent and tp_percent columns in production_setting, divide the values by 10
        production_setting['sl_percent'] = production_setting['sl_percent'] / 10
        production_setting['tp_percent'] = production_setting['tp_percent'] / 10
        
        # save the production setting to a csv file
        folder = "./vbt_pro/RSI_Cross/" # Folder to store all optimised setting
        file_name = f'RSI_Cross - Production Setting {datetime.now().strftime("%Y.%m.%d %H.%M.%S")}.csv'
        production_setting.to_csv(folder + file_name)
    
    if(save_plot):
        # save the plot to a png file
        folder = "./vbt_pro/RSI_Cross/" # Folder to store all optimised setting
        file_name = f'RSI_Cross - Production IS equity {datetime.now().strftime("%Y.%m.%d %H.%M.%S")}.png'
        fig.savefig(folder + file_name)


# %% HyperParameter Tuning with HyperOpt
#fast = 2 - 35
# gap, fast + gap = slow = 0 - 20

# threshold = 15 - 40
#market non neurality = -10 - 10
#rsi_thresh_entry = threshold + market non neurality
#rsi_thresh_exit = 100 - threshold
settings_param = {'rsi_fast_period': hp.quniform('rsi_fast_period', 5, 41, 2),
                  'rsi_slow_period': hp.quniform('rsi_slow_period', 5, 41, 2),
                  'rsi_thresh_entry': hp.quniform('rsi_thresh_entry', 10, 45, 2),
                  'rsi_thresh_exit': hp.quniform('rsi_thresh_exit', 55, 90, 2),
                  'rsi_candle': hp.randint('rsi_candle', 0, 6),
                  'rsi_cross_entry': hp.randint('rsi_cross_entry', 1, 5), 
                  'rsi_cross_exit': hp.randint('rsi_cross_exit', 1, 3),
                  'max_trade': hp.randint('max_trade', 1, 8),
                  'tp_percent': hp.quniform('tp_percent', 5, 55, 2),
                  'sl_percent': hp.quniform('sl_percent', 2, 25, 2),
                  }
        


#%%
file_path = r"C:/Users/user/Downloads/AUDUSD_GMT+2_US-DST 01.01.2019 to 20.02.2023_M5.csv".replace('\\', '/')
price_df = read_csv_to_df(file_path=file_path)
# price_df = price_df.iloc[:130000]
# price_df = price_df.iloc[-60000:]

# open_price = price_df.get('Open')
# close_price = price_df.get('Close')
# high_price = price_df.get('High')
# low_price = price_df.get('Low')
# price = open_price

#%% local data import

# AUDUSD_5m = r"C:/Users/user/Downloads/AUDUSD_GMT+2_US-DST 01.01.2019 to 20.02.2023_M5.csv".replace('\\', '/')
# AUDUSD_5m = read_csv_to_df(file_path=AUDUSD_5m)
# USDCAD_5m = r"C:/Users/user/Downloads/USDCAD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# USDCAD_5m = read_csv_to_df(file_path=USDCAD_5m)
# GBPUSD_5m = r"C:/Users/user/Downloads/GBPUSD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# GBPUSD_5m = read_csv_to_df(file_path=GBPUSD_5m)
# EURUSD_5m = r"C:/Users/user/Downloads/EURUSD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# EURUSD_5m = read_csv_to_df(file_path=EURUSD_5m)
# USDCHF_5m = r"C:/Users/user/Downloads/USDCHF_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# USDCHF_5m = read_csv_to_df(file_path=USDCHF_5m)
XAUUSD_5m = r"C:/Users/user/Downloads/XAUUSD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
XAUUSD_5m = read_csv_to_df(file_path=XAUUSD_5m)
WTI_5m = r"C:/Users/user/Downloads/US_Light_Crude_Oil_GMT+2_US-DST  01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
WTI_5m = read_csv_to_df(file_path=WTI_5m)
BRENT_5m = r"C:/Users/user/Downloads/US_Brent_Crude_Oil_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
BRENT_5m = read_csv_to_df(file_path=BRENT_5m)
# US30_5m = r"C:/Users/user/Downloads/USA_30_Index_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# US30_5m = read_csv_to_df(file_path=US30_5m)
# US500_5m = r"C:/Users/user/Downloads/USA_500_Index_GMT+2_US-DST 01.01.2019 to 28.02.2023_M5.csv".replace('\\', '/')
# US500_5m = read_csv_to_df(file_path=US500_5m)




# local data import for 30 min timeframe
# USDCAD_30 = r"C:/Users/user/Downloads/USDCAD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M30.csv".replace('\\', '/')
# USDCAD_30 = read_csv_to_df(file_path=USDCAD_30)
# EURUSD_30 = r"C:/Users/user/Downloads/EURUSD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M30.csv".replace('\\', '/')
# EURUSD_30 = read_csv_to_df(file_path=EURUSD_30)
# GBPUSD_30 = r"C:/Users/user/Downloads/GBPUSD_GMT+2_US-DST 01.01.2019 to 28.02.2023_M30.csv".replace('\\', '/')
# GBPUSD_30 = read_csv_to_df(file_path=GBPUSD_30)


#%%
# trim data to the same start and end date for all dataframes
raw_dict = {'XAUUSD_5m' : XAUUSD_5m, 'WTI_5m' : WTI_5m, 'BRENT_5m' : BRENT_5m}#, 'USDCHF' : USDCHF}
vbt_data = trim_data(raw_dict, start='2022-7-28', period=3)
vbt_data_oos = trim_data(raw_dict, start='2022-10-28', period=3)
#%% do Hyperopt for all the instruments, with same max_eval & Top N
best_setting_dict = hyperopt_factory(vbt_data, max_evals=1000, top_N=100)
#%% get the pnl (cumsum of returns) for each settings for each instruement
pnl_dict = best_settings_dict_to_pnl(best_setting_dict, vbt_data)
#%% monte carlo simulation to get the pnl plot for each instrument
monte_carlo(pnl_dict)
#%% oos
# pnl_dict_oos = best_settings_dict_to_pnl(best_setting_dict, vbt_data_oos)
# monte_carlo(pnl_dict_oos)
#%% use the pnl to perform corelation analysis
corr_matrix = pnl_to_corr_matrix(pnl_dict)
#%% get the settings (symbol + tf + setting id) for each sub matrix
portfolio_symbols = corr_matrix_to_portfolio_symbols(corr_matrix, vbt_data, 100)
#%% do the backbest (IS) to visualize the performance historically, to_csv means save the setting for production use, save plot is to save png
portfolio_backtest(portfolio_symbols, best_setting_dict, vbt_data, to_csv=True, save_plot=True, title='In Sample')
#%%
portfolio_backtest(portfolio_symbols, best_setting_dict, vbt_data_oos, to_csv=True, save_plot=True, title='Out of Sample')




# #%%
# # testing
# s1 = pd.Series([5, 8, 2, 5], index=[1, 2, 3, 4])
# s2 = pd.Series([7, 12, 8, 3], index=[2, 3, 5, 8])
# s3 = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 10])

# df = pd.concat([s1, s2, s3], axis=1)

# fig, ax = plt.subplots()
# s1.plot(ax=ax, label="s1")
# s2.plot(ax=ax, label="s2")
# s3.plot(ax=ax, label="s3")
# ax.legend()


# #%% bt_result_trial top 10 best loss

# loss = []
# for trials in bt_result_trial:
#     if (trials.get('status') == 'fail'):
#         loss.append(20)
#     else:
#         loss.append(round(trials.get('loss'), 2))
# loss_series = pd.Series(loss)
# loss_series_filtered = loss_series.drop_duplicates()
# loss_series_filtered = loss_series_filtered.nsmallest(100)

# #%%

# # create a list variable to store the data to reconstruct the plot for all 100 best loss

# is_list = []
# oos_list = []

# for i, index in enumerate(loss_series_filtered.index):

#     best_params = bt_result_trial[index].get("Setting")
#     res_bt = rsi_bt_ho_ohlc(is_price, best_params)
#     pf = res_bt['PF']
#     # Get the setting into Series
#     setting_set = pd.Series(best_params, index=settings_param.keys())
#     setting_set = setting_set.rename(index={'rsi_slow_period': 'gap', 'rsi_thresh_exit': 'non neutrality'})
#     trades = pf.trades.records_readable
#     trades_plot = trades.set_index('Exit Index')
#     is_trade_plot = trades_plot
#     res_bt = rsi_bt_ho_ohlc(oos_price, best_params)
#     res_bt['Symbol'] = Acc_symbol.replace(':', '_')  # Store symbol
#     res_bt['Timeframe'] = Acc_tf  # Store timeframe
#     res_bt['Sample'] = "Out of Sample"
#     # Get the setting into Series
#     # bt_result = bt_result.append(res_bt, ignore_index=True)  # Save into DF

#     # -- Plot equity graph from best
#     pf = res_bt['PF']
#     # save the sharpe ratio, sortino ratio, and the number of trades, and the max drawdown duration to a list
#     sharpe_ratio = pf.sharpe_ratio.mean()
#     sortino_ratio = pf.sortino_ratio.mean()
#     total_trades = (pf.trades.records_readable['Status'] == 'Closed').sum().min()
#     max_drawdown_duration = pf.stats(metrics="max_dd_duration")[0]
#     #convert the timedelta to an integer
#     max_drawdown_duration_int = pf.stats(metrics="max_dd_duration")[0].total_seconds() / 86400
#     top_N = i + 1
#     # save all these metrics to a list then convert to pandas series
#     metrics = [sharpe_ratio, sortino_ratio, total_trades, max_drawdown_duration, top_N]
#     metrics = pd.Series(metrics, index=['Sharpe Ratio', 'Sortino Ratio', 'Total Trades', 'Max Drawdown Duration', "top_N"])

#     trades = pf.trades.records_readable
#     trades_plot = trades.set_index('Exit Index')
#     oos_trade_plot = trades_plot
#     # ##### OOS Sample ######

#     ## Now need to plot the in sample and out of sample side by side

#     is_trade_plot['IS PnL'] = is_trade_plot.groupby('Column')['PnL'].cumsum()
#     oos_trade_plot['OOS PnL'] = oos_trade_plot.groupby('Column')['PnL'].cumsum()
#     is_list.append(is_trade_plot['IS PnL'])
#     oos_list.append(oos_trade_plot['OOS PnL'])
#     plt.figure(figsize=(14, 6))
#     plt.plot(is_trade_plot.index, is_trade_plot['IS PnL'], color='blue', label='IS PnL')
#     plt.plot(oos_trade_plot.index, oos_trade_plot['OOS PnL'], color='orange', label='OOS PnL')
#     plt.xlabel('Exit index')
#     plt.xticks(rotation=90)
#     plt.ylabel('PnL')
#     plt.legend(loc='upper left')
#     # append i to the setting_set
#     plt.figtext(0.7, 0.25, metrics.to_string())
#     plt.title(setting_set.to_string())
#     plt.show()
    
    
# #%% outer join all the pnl series to find the mean of the monte carlo simulation

# is_list

# start_date = min(s.index.min() for s in is_list)
# end_date = max(s.index.max() for s in is_list)
# is_date_range = pd.date_range(start=start_date, end=end_date, freq='T')
# df = pd.DataFrame(index=is_date_range)


# for s in is_list:
    
#     # drop duplicated index
#     s = s[~s.index.duplicated(keep='first')]
#     s = s.resample(rule='M').mean()
#     df = pd.concat([df, s], axis=1, join='outer')
# df = df.dropna(how='all')
    
# df = df.fillna(method='ffill').fillna(method='bfill')
    
# is_mean = df.mean(axis=1)
# # convert 1 column df to series
# is_mean = is_mean.squeeze()
# # is_mean = is_mean.resample(rule='D').sum()


# start_date = min(s.index.min() for s in oos_list)
# end_date = max(s.index.max() for s in oos_list)
# oos_date_range = pd.date_range(start=start_date, end=end_date, freq='T')
# df = pd.DataFrame(index=oos_date_range)


# for i, s in enumerate(oos_list):
#     s = s[~s.index.duplicated(keep='first')]
#     s = s.resample(rule='M').mean()
#     df = pd.concat([df, s], axis=1, join='outer')
# df = df.dropna(how='all')
    
# df = df.fillna(method='ffill').fillna(method='bfill')
    
# oos_mean = df.mean(axis=1)
# # convert 1 column df to series
# oos_mean = oos_mean.squeeze()
# # oos_mean = oos_mean.resample(rule='D').sum()
# #%%
# # Overlay the 100 plots on top of each other
# # plt.figure(figsize=(14, 6))
# # for i in range(100):
# #     plt.plot(is_list[i], color='blue')#, label='IS PnL')
# #     plt.plot(oos_list[i], color='orange')#, label='OOS PnL')
# # plt.xlabel('Exit index')
# # plt.xticks(rotation=90)
# # plt.ylabel('PnL')
# # plt.legend(loc='upper left')
# # plt.title("Overlay of the 100 best loss")
# # plt.show()


# #%% sum up all the plot to 1 monte carlo simulation with a black avg line
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# # Plot IS data on first subplot
# for i in range(20):
#     ax1.plot(is_list[i], color='blue', alpha=0.5)# label='IS PnL')
# ax1.plot(is_mean, color='black')
# ax1.set_xlabel('Exit index')
# ax1.set_xticklabels(is_mean.index, rotation=30)
# ax1.set_ylabel('PnL')
# ax1.set_title("IS PnL")

# # Plot OOS data on second subplot
# for i in range(20):
#     ax2.plot(oos_list[i], color='orange', alpha=0.5)#, label='OOS PnL')
# ax2.plot(oos_mean, color='black')
# ax2.set_xlabel('Exit index')
# ax2.set_xticklabels(oos_mean.index, rotation=90)
# ax2.set_ylabel('PnL')
# ax2.set_title("OOS PnL")

# plt.show()




# %%
