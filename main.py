#!/usr/bin/python

# coding: utf-8
# Running the entire model

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cluster import KMeans
from skimage.feature import hog
import math
import pandas as pd

from evaluation import *
from basic_processing import *
from bayesian_regression import *

def one_period(price, price2, price_val, price_val_src, cash_balance):

    len1 = len(price)
    len2 = len(price2)

    # For AAPL
    ts1 = generate_timeseries(price[:len1/2], ts1len, time_window)
    ts2 = generate_timeseries(price[:len1/2], ts2len, time_window)
    ts3 = generate_timeseries(price[:len1/2], ts3len, time_window)
    # For GOOG
    ts4 = generate_timeseries(price2[:len2/2], ts1len, time_window)
    ts5 = generate_timeseries(price2[:len2/2], ts2len, time_window)
    ts6 = generate_timeseries(price2[:len2/2], ts3len, time_window)

    print("Generation of timeseries done.")

    # For AAPL
    centers1 = find_cluster_centers(ts1[0], ts1[1], nkmean)
    centers2 = find_cluster_centers(ts2[0], ts2[1], nkmean)
    centers3 = find_cluster_centers(ts3[0], ts3[1], nkmean)
    # For GOOG
    centers4 = find_cluster_centers(ts4[0], ts4[1], nkmean)
    centers5 = find_cluster_centers(ts5[0], ts5[1], nkmean)
    centers6 = find_cluster_centers(ts6[0], ts6[1], nkmean)

    print("%d cluster centers found." % nkmean)

    s1 = choose_effective_centers(centers1, ncenter, m_top_discard)
    s2 = choose_effective_centers(centers2, ncenter, m_top_discard)
    s3 = choose_effective_centers(centers3, ncenter, m_top_discard)

    s4 = choose_effective_centers(centers4, ncenter, m_top_discard)
    s5 = choose_effective_centers(centers5, ncenter, m_top_discard)
    s6 = choose_effective_centers(centers6, ncenter, m_top_discard)

    print("%d cluster centers selected." % ncenter)
#    print("s1, s2, s3", s1, s2, s3)

    Dpi_r, Dp = linear_regression_vars(price[len1/2:], s1, s2, s3, s4, s5, s6, time_window)
    print("Bayesian regression done.")

    w = find_w_linear_regression(Dpi_r, Dp)
#    print("Linear regression done.", w)

#    dps, pprice = predict_dps(price, s1, s2, s3, w, time_window)
    print("Done training. Start Validation.")
    dps_val, pprice_val = predict_dps(price_val, s1, s2, s3, s4, s5, s6, w, time_window)

    dprice = diff_price(price_val)
    mdprice = mean_price(dprice, 20)

#    plt.plot(dps_val)
#    plt.plot(pprice_val)
#    plt.plot(dprice)
    #plt.plot(mdprice)
#    plt.plot(price_val)
#    plt.show()

#    plt.plot(dprice[ts3len+time_window:])
#    plt.plot(dps_val)
#    plt.show()

    # evaluation 1 -- original
#    bank_balance = evaluate_performance(price, dps, ts3len, time_window, t=0.001, step=1)
#    print("Final Balance:", bank_balance)

    # evaluation 2 -- track the change of total asset
    asset = evaluate_performance_asset(price_val_src, dps_val, ts3len, time_window, 0.005, 1, cash_balance)
#    asset = evaluate_performance_asset_moving(price_val_src, dps_val, ts3len, time_window, 0.001, smooth_window, cash_balance)

#    plt.plot(asset/10.0)
#    plt.plot(collapse(price_val[ts3len:], time_window))
#    plt.plot(collapse(price_val_src[ts3len:], time_window))
    return pprice_val, asset


def main():
    one_stock = io_price("AAPL2")
    price_init = collapse(one_stock, time_intval)[offset:offset+npts+nperiod*nval]
    # shift by 500 to skip the big dip at the begining of source data
    two_stock = io_price("AAPL")
    price2_init = collapse(two_stock, time_intval)[offset-ts3len:offset-ts3len+npts+nperiod*nval]

    price_src = price_init[smooth_window:]

    price_init = smooth_price(price_init, smooth_window)
    price2_init = smooth_price(price2_init, smooth_window)

    # Outer loops
    all_asset = []
    all_pprice = []
    all_price = []
    cash_balance = price_init[0]

    for i in range(nperiod):
        price = price_init[i*nval:npts-nval+i*nval]
        price2 = price2_init[i*nval:npts-nval+i*nval]
	
        price_val = price_init[i*nval+npts-nval-ts3len:npts+i*nval]
        price_val_src = price_src[i*nval+npts-nval-ts3len:npts+i*nval]

        pprice, asset = one_period(price, price2, price_val, price_val, cash_balance)  # change the second price_val to price_val_src to use real price
        cash_balance = asset[-1]

        all_asset.append(asset)
        all_pprice.append(pprice[ts3len+time_window:])
        all_price.append(price_val[ts3len+time_window:])


    all_asset = np.array(all_asset).flatten()
    all_pprice = np.array(all_pprice).flatten()
    all_price = np.array(all_price).flatten()
    
    plt.plot(all_price)
    plt.plot(all_pprice)
    plt.plot(price_src[npts-nval:npts-nval+len(all_asset)])
    plt.plot(all_asset + all_price[0] - all_asset[0])

    plt.title('Total asset')
    plt.show()
    print('Start Asset: %f' % all_asset[0])
    print('Final Asset: %f' % all_asset[-1])



if __name__ == '__main__':
    time_intval = 1.0 # basic unit
    time_window = 5 # in time_intval unit
    npts = 1000  # including nval, and the rest: half to find centers, half to do linear regression
    nval = 200
    smooth_window = time_window
    nperiod = 5
    ts1len, ts2len, ts3len = 3, 10, 20
    offset = ts3len + 10000
    nkmean, ncenter, m_top_discard = 100, 60, 1 # discard top m centers, because these are likely rare
    main()
