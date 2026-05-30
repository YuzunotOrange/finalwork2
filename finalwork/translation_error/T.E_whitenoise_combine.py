import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import random

def compute_average_medians(series, N_0_repeat=50, kinbou=4, A_repeat=10, tau=5, m_range=range(1, 11)):
    average_medians = []
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
    return average_medians

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", help="path of analysis csv file")
    args = parser.parse_args()

    # Analysis parameters
    N_0_repeat = 50
    kinbou = 4
    A_repeat = 10
    tau = 5
    m_range = range(1, 11)

    # Load data
    new_data = pd.read_csv(args.csv_files, header=None, delim_whitespace=True)
    series_orig = new_data.iloc[:, 0].values

    # Noise-free
    print("\n===== Processing NOISE-FREE data =====")
    average_medians_clean = compute_average_medians(
        series_orig,
        N_0_repeat=N_0_repeat,
        kinbou=kinbou,
        A_repeat=A_repeat,
        tau=tau,
        m_range=m_range
    )

    # Add noise
    uniform_noise = np.random.uniform(0, 1, size=series_orig.shape)
    series_noisy = series_orig + uniform_noise

    print("\n===== Processing NOISY data =====")
    average_medians_noisy = compute_average_medians(
        series_noisy,
        N_0_repeat=N_0_repeat,
        kinbou=kinbou,
        A_repeat=A_repeat,
        tau=tau,
        m_range=m_range
    )
    print("\n===== Translation Error Values =====")
    print("Embedding dimension m | Noise-free | With noise")
    for m_val, clean, noisy in zip(m_range, average_medians_clean, average_medians_noisy):
        print(f"m={m_val:2d}  | {clean:.6f} | {noisy:.6f}")


    # Plot both
    plt.figure(figsize=(10, 6))
    plt.plot(list(m_range), average_medians_clean, marker='o', label='Noise-free')
    plt.plot(list(m_range), average_medians_noisy, marker='s', label='With noise')
    plt.xlabel('Embedding dimension m')
    plt.ylabel('Translation error')
    plt.yscale('log')
    plt.title(f'Translation Error vs Embedding Dimension (tau={tau})')
    plt.xticks(list(m_range))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
