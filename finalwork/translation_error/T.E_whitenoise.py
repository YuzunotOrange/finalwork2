import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import random


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

    # ホワイトノイズ（0〜1の一様分布）を加える
    uniform_noise = np.random.uniform(0, 1, size=series.shape)
    series = series + uniform_noise


    tau = 5

    max_m = 11
    min_m = 2
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
    plt.title(f'Translation Error vs Embedding Dimension (t={tau})')
    plt.xticks(list(m_range))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    t2 = time.time()
    print("Complete")
    print("Processing time:", time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))

if __name__ == "__main__":
    main()