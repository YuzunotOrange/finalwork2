import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import time
import random
import glob
import argparse
import shutil


def calculate_autocorrelation(series, max_lag):
    autocorr_values = []
    series_mean = np.mean(series)
    for lag in range(1, max_lag + 1):
        numerator = np.sum((series[:-lag] - series_mean) * (series[lag:] - series_mean))
        denominator = np.sum((series - series_mean) ** 2)
        autocorr = numerator / denominator if denominator != 0 else 0.0
        autocorr_values.append(autocorr)
    return autocorr_values

def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threshold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1
    return max_lag


def run_method1(series, N_0_repeat = 50, kinbou = 4, A_repeat = 10):
    print(f"\n==== [Method 1] Start ====")
    
    series = (series - np.mean(series)) / np.std(series)
    time_delay = determine_tau(series)

    print(f"[INFO] τ = {time_delay}")

    max_m = 10
    min_m = 1
    m_range = range(min_m, max_m + 1)
    average_medians = []

    t1 = time.time()

    for m in m_range:
        print(f"--- embedding dimension m = {m} ---")
        length = len(series) - m * time_delay
        if length <= 1:
            print(f"[WARNING] Skipping m = {m} because length={length} <= 0")
            average_medians.append(np.nan)
            continue

        embedded_vectors = np.array([
            [series[i + j * time_delay] for j in range(m)]
            for i in range(length)
        ])

        data_tate = embedded_vectors.shape[0]
        if data_tate < kinbou + 2:
            print(f"[WARNING] Skipping m = {m} because not enough embedded vectors (data_tate = {data_tate})")
            average_medians.append(np.nan)
            continue

        median_list = []
        for _ in range(A_repeat):
            e_trans_list = []
            for _ in range(N_0_repeat):
                if data_tate < 2:
                    break  # skip if not enough data for referencing

                ref_max = data_tate - 2
                if ref_max < 0:
                    continue
                ref = random.randint(0, ref_max)

                dists = np.linalg.norm(embedded_vectors - embedded_vectors[ref], axis=1)
                nearest_indices = np.array([i for i in np.argsort(dists) if i + 1 < data_tate][:kinbou + 1])
                if len(nearest_indices) < kinbou + 1:
                    continue

                x = embedded_vectors[nearest_indices]
                y = embedded_vectors[nearest_indices + 1]
                v = y - x
                v_ave = np.mean(v, axis=0)
                bunbo = np.sum(v_ave ** 2)
                e_trans = 0 if bunbo == 0 else np.sum([(np.sum((vi - v_ave) ** 2)) / bunbo for vi in v]) / (kinbou + 1)
                e_trans_list.append(e_trans)
            if len(e_trans_list) > 0:
                median_list.append(np.median(e_trans_list))

        mean_medians = np.mean(median_list) if len(median_list) > 0 else np.nan
        average_medians.append(mean_medians)
    return list(m_range), average_medians

def run_method2(series, tau = 5, N_0_repeat = 50, kinbou = 4, A_repeat = 10, add_noise=False):
    if add_noise:
        print("[INFO] Adding uniform noise to series in Method 2")
        uniform_noise = np.random.uniform(0, 1, size=series.shape)
        series = series + uniform_noise

    
    max_m = 11
    min_m = 1

    m_range = range(min_m, max_m + 1)
    average_medians = []

    t1 = time.time()

    m_range = range(1, 11)
    for m in m_range:
        print(f"\n<<<<embedding dimension m = {m} >>>")
        embedded_vectors = []
        
        for i in range(len(series) - m * tau):
            embedded = [series[i + j * tau] for j in range(m)]
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

    diff = np.diff(average_medians)
    max_diff_index = np.argmax(np.abs(diff))
    start = max(0, max_diff_index - 1)
    end = min(len(average_medians), max_diff_index + 2)
    threadhold = np.mean(average_medians[start:end])
    return list(m_range), average_medians

# ----------------- Main -----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("method2_csv", help="Path to CSV for Method 2 (e.g. lorenz.csv)")
    parser.add_argument("--input_dir", default="new_li", help="Directory with CSVs for Method 1")
    parser.add_argument("--output_dir", default="finalresult", help="Directory to save plots")
    args = parser.parse_args()

    # Make output dir
    import shutil
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    print(f"[INFO] Created fresh output directory: {args.output_dir}")


    # Load Method 2 data once
    print("\n==== Computing Method 2 (fixed τ=5) ====")
    series2 = pd.read_csv(args.method2_csv, header=None, delim_whitespace=True).iloc[:, 0].values
    
    #Method 2 without noise
    m2_no_noise, avg2_no_noise = run_method2(series2, add_noise=False)

    #Method 2 with noise
    m2_with_noise, avg2_with_noise = run_method2(series2, add_noise=True)
    # For each file in new_li, compute Method 1 and plot both
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"\n==== Processing {filename} for Method 1 ====")
        series1 = pd.read_csv(filepath, header=None, delim_whitespace=True).iloc[:, 0].values
        m1, avg1 = run_method1(series1)

        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(m1, avg1, marker='o', label=f'Method 1 ({filename})')
        plt.plot(m2_no_noise, avg2_no_noise, marker='s', label=f'Method 2 (No Noise)')
        plt.plot(m2_with_noise, avg2_with_noise, marker='^', label=f'Method 2 (With Noise)')
        plt.xlabel('Embedding Dimension m')
        plt.ylabel('Translation Error')
        plt.yscale('log')
        plt.title(f'Comparison: {filename} vs {os.path.basename(args.method2_csv)}')
        plt.legend()
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.xticks(range(1, 11))
        plt.tight_layout()
        savepath = os.path.join(args.output_dir, f"{filename}_comparison.png")
        plt.savefig(savepath)
        plt.close()
        print(f"[Saved] {savepath}")

    print("\n[Done] All comparisons finished.")

if __name__ == "__main__":
    main()