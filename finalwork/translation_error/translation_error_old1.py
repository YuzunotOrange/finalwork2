import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random
import glob

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

def process_file(filepath, output_dir):
    filename = os.path.basename(filepath).split('.')[0]
    print(f"\n==== Processing: {filename} ====")

    N_0_repeat = 50
    kinbou = 4
    A_repeat = 10

    new_data = pd.read_csv(filepath, header=None, delim_whitespace=True)
    series = new_data.iloc[:, 0].values
    series = (series - np.mean(series)) / np.std(series)

    time_delay = determine_tau(series)
    print(f"[INFO] τ = {time_delay}")

    max_m = 11
    min_m = 2
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

    # Post-process
    diff = np.diff(average_medians)
    max_diff_index = np.nanargmax(np.abs(diff)) if not np.all(np.isnan(diff)) else 0
    start = max(0, max_diff_index - 1)
    end = min(len(average_medians), max_diff_index + 2)
    threshold = np.nanmean(average_medians[start:end])
    print(f"[RESULT] Estimated threshold for {filename}: {threshold:.4f}")
    print(f"[INFO] Range of m which max change: m = {start + 1} ~ {end}")
    print(f"[INFO] Average of translation error in this range(radius): {threshold:.4f}")

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(m_range), average_medians, marker='o')
    plt.xticks(list(m_range))
    plt.xlabel('Embedding Dimension m')
    plt.ylabel('Translation Error log')
    plt.yscale('log')
    plt.title(f'Translation Error vs Embedding Dimension\n({filename}) τ={time_delay}')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

    # Save result text
    with open(os.path.join(output_dir, f"{filename}_result.txt"), 'w') as f:
        f.write(f"τ = {time_delay}\n")
        f.write("Translation error (mean median):\n")
        for m, val in zip(m_range, average_medians):
            f.write(f"m={m}: {val:.4f}\n")
        f.write(f"\nRange of significant change: m = {start + 1} ~ {end}\n")
        f.write(f"Radius average: {threshold:.4f}\n")

    t2 = time.time()
    print(f"[DONE] {filename} in {time.strftime('%H:%M:%S', time.gmtime(t2 - t1))}")

def main():
    input_dir = "new_li"
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in 'new_li' directory.")
        return

    for file in csv_files:
        try:
            process_file(file, output_dir)
        except Exception as e:
            print(f"[ERROR] Failed processing {file}: {e}")

if __name__ == "__main__":
    main()
