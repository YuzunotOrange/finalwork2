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

    #Analysis condition
    N_0_repeat = 50
    kinbou = 4
    A_repeat = 10

    #scan data
    new_data = pd.read_csv(args.csv_files, header=None, delim_whitespace=True)
    series = new_data.iloc[:, 0].values
    series = (series - np.mean(series)) / np.std(series)

    #time_delay = determine_time_delay_zero_crossing(series)
    time_delay = determine_tau(series)
    #time_delay = determine_tau_first_min(series)
    
    print(f"time_delay (first lag where autocorr <= 0): {time_delay}")

    max_m = 11
    min_m = 2
    m_range = range(min_m, max_m+1)
    average_medians = []

    t1 = time.time()

    m_range = range(1, 10)
    for m in m_range:
        print(f"\n<<<<embedding dimension m = {m} >>>")
        embedded_vectors = []
        
        for i in range(len(series) - m * time_delay):
            embedded = [series[i + j * time_delay] for j in range(m)]
            embedded_vectors.append(embedded)

        embedded_vectors = np.array(embedded_vectors)
        data_tate, data_yoko = embedded_vectors.shape

        median_list = []
        for _ in range(A_repeat):
            e_trans_list = []
            for _ in range(N_0_repeat):
                ref = random.randint(0, data_tate - 2)
                dists = np.linalg.norm(embedded_vectors - embedded_vectors[ref], axis=1)
                nearest_indices = np.array([i for i in np.argsort(dists) if i + 1 < data_tate][:kinbou + 1])
                x = embedded_vectors[nearest_indices]
                y = embedded_vectors[nearest_indices + 1]
                v = y - x
                v_ave = np.mean(v, axis=0)
                
                bunbo = np.sum(v_ave ** 2)
                if bunbo == 0:
                    e_trans = 0
                else:
                    sigma = np.sum([np.sum((vi - v_ave) ** 2) / bunbo for vi in v])
                    e_trans = sigma / (kinbou + 1)
                    
                e_trans_list.append(e_trans)
                
            median_list.append(np.median(e_trans_list))
        mean_medians = np.mean(median_list)
        average_medians.append(mean_medians)
    print(f"m = {m} avrage translation error number of center: {average_medians}")

    diff = np.diff(average_medians)
    max_diff_index = np.argmax(np.abs(diff))
    start = max(0, max_diff_index - 1)
    end = min(len(average_medians), max_diff_index + 2)
    threadhold = np.mean(average_medians[start:end])

    print(f"\n[INFO] Range of m which max change: m = {start + 1} ~ {end}")
    print(f"[INFO] Average of translation error in this range(radius): {threadhold:.4f}")
    #graph plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(m_range), average_medians, marker = 'o')
    plt.xlabel('embedded dimension m')
    plt.ylabel('translation error')
    plt.title(f'Translation Error vs Embedding Dimension (t={time_delay})')
    plt.xticks(list(m_range))
    #autocorr = calculate_autocorrelation(series, max_lag=100)
    #plt.plot(range(1, 101), autocorr)
    #plt.axhline(1/np.e, color='red', linestyle='--')
    #plt.title('Autocorrelation')
    #plt.xlabel('Lag')
    #plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    t2 = time.time()
    print("Complete")
    print("Processing time:", time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))

if __name__ == "__main__":
    main()