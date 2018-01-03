# This module contains the evaluation functions of the prediction model
# Update: 04/19/2017

import numpy as np

def evaluate_performance(price, dps, ts3len, time_window, t, step):
    bank_balance = 0
    position = 0
    for i in range(ts3len, len(price) - time_window, step):
        # long position - BUY
        if dps[i - ts3len] > t and position <= 0:
            position += 1
            bank_balance -= price[i]
        # short position - SELL
        if dps[i - ts3len] < -t and position >= 0:
            position -= 1
            bank_balance += price[i]
    # sell what you bought
    if position == 1:
        bank_balance += price[len(price) - 1]
    # pay back what you borrowed
    if position == -1:
        bank_balance -= price[len(price) - 1]
    return bank_balance

def evaluate_performance_asset(price, dps, ts3len, time_window, t, step, cash_balance):
    nShare = 0
#    cash_balance = price[ts3len]*10.0
    cash_balance *= 10.0
    all_asset = []
    for i in range(ts3len, len(price) - time_window, step):
        # long position - BUY
        if dps[i - ts3len] > t and cash_balance > 0:
            nBuy = np.floor(cash_balance/price[i+time_window-1])
            nShare += nBuy
            if nBuy > 0:
                print('Time ', i, 'Buy , ', nBuy, ' #Share, ', nShare)
            cash_balance -= price[i+time_window-1]*nBuy
        # short position - SELL
        if dps[i - ts3len] < -t and nShare > 0:
            nSell = nShare
            cash_balance += price[i+time_window-1] * nSell
            nShare = 0
            if nSell > 0:
                print('Time ', i, 'Sell, ', nSell, ' #Share, ', nShare)
        all_asset.append(cash_balance + nShare*price[i+time_window-1])
    return np.array(all_asset)/10.0

def evaluate_performance_asset_moving(price, dps, ts3len, time_window, t, step, cash_balance):
    nShare = 0
#    cash_balance = price[ts3len]*10.0
    cash_balance *= 10.0
    all_asset = []
    for i in range(ts3len, len(price) - time_window, 1):
        price_to_use = np.mean(price[i:i+step])
        # long position - BUY
        if dps[i - ts3len] > t and cash_balance > 0:
            nBuy = np.floor(cash_balance/price_to_use)
            nShare += nBuy
            if nBuy > 0:
                print('Time ', i, 'Buy , ', nBuy, ' #Share, ', nShare)
            cash_balance -= price_to_use*nBuy
        # short position - SELL
        if dps[i - ts3len] < -t and nShare > 0:
            nSell = nShare
            cash_balance += price_to_use * nSell
            nShare = 0
            if nSell > 0:
                print('Time ', i, 'Sell, ', nSell, ' #Share, ', nShare)
        all_asset.append(cash_balance + nShare*price_to_use)
    return np.array(all_asset)/10.0


