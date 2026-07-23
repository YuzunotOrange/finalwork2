import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import random
import numpy as np

#相互情報量における計算をしてる
def mutual_information(X, Y, bins=30):
    p_xy, x_edges, y_edges = np.histogram2d(X, Y, bins=bins, density=True)
    p_x, _ = np.histogram(X, bins=x_edges, density=True)
    p_y, _ = np.histogram(Y, bins=y_edges, density=True)
    p_x_y = p_x[:, None] * p_y[None, :]

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p_xy > 0, p_xy / p_x_y, 1)
        elem = p_xy * np.log(ratio)
    mi = np.nansum(elem) * dx * dy
    return mi

def determine_tau_mutual_info(series, max_lag=100):
    mi_values = []
    for lag in range(1, max_lag + 1):
        Y = np.roll(series, -lag)[:-lag]
        X = series[:-lag]
        mi = mutual_information(X, Y)
        mi_values.append(mi)
    tau = np.argmin(mi_values) + 1  # 最小相互情報量の遅れ時間を選択
    return tau, mi_values


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
    time_delay, mi_values = determine_tau_mutual_info(series)
    #time_delay = determine_tau_first_min(series)
    
    print(f"time_delay(mutal info): {time_delay}")

    max_m = 11
    min_m = 2
    m_range = range(min_m, max_m+1)
    average_medians = []

    t1 = time.time()

    m_range = range(1, 11)
    vaild_m = []
    average_medians = []

    for m in m_range:
        length = len(series) - m * time_delay
        if length <= 0:
            print(f"[WARNING] Not enough data points for m={m}, time_delay={time_delay}. Skipping...")
            continue  # または break など適宜処理
        
        embedded_vectors = []
        for i in range(length):
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
        vaild_m.append(m)
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
    plt.plot(vaild_m, average_medians, marker = 'o')
    plt.xlabel('embedded dimension m')
    plt.ylabel('translation error log')
    plt.yscale('log')
    plt.title(f'Translation Error vs Embedding Dimension (Mutual Information) (t={time_delay})')
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