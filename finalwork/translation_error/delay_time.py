import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import random

def calculate_autocorrelation(series, max_lag):
    """
    時系列データの自己相関関数を計算し、
    指定された最大遅れ時間までの自己相関値をリストで返します。
    """
    autocorr_values = []
    n = len(series)
    series_mean = np.mean(series)

    for lag in range(1, max_lag + 1):
        numerator = np.sum((series[:-lag] - series_mean) *
                           (series[lag:] - series_mean))
        denominator = np.sum((series - series_mean) ** 2)
        
        if denominator == 0:
            autocorr = 0.0
        else:
            autocorr = numerator / denominator
        autocorr_values.append(autocorr)
    return autocorr_values

#def determine_time_delay_zero_crossing(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    for i, val in enumerate(autocorr):
        if val <= 0:
            return i + 1 #log start from 1
    return max_lag #0以下が見つからなければmax_lagを返す
def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threadhold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threadhold:
            return i + 1 #log start from 1
    return max_lag #fallback

#def determine_tau_first_min(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    for i in range(1, len(autocorr)-1):
        if autocorr[1] < autocorr[i - 1] and autocorr[i] < autocorr[i+1]:
            return i + 1
    return max_lag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", help="pass of analysis csv pass")
    args = parser.parse_args()

    #scan data
    new_data = pd.read_csv(args.csv_files, header=None, delim_whitespace=True)
    series = new_data.iloc[:, 0].values
    series = (series - np.mean(series)) / np.std(series)

    #time_delay = determine_time_delay_zero_crossing(series)
    time_delay = determine_tau(series)
    #time_delay = determine_tau_first_min(series)
    
    print(f"time_delay (first lag where autocorr <= 0): {time_delay}")

if __name__ == "__main__":
    main()