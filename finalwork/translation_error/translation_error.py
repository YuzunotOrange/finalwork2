import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal, ROUND_HALF_UP
from random import SystemRandom
import random

# 適用時の条件設定
N_0_repeat = 1000  # 基準ベクトルの選択個数 N_0
kinbou = 4          # 抽出する近接ベクトル数 k
A_repeat = 10       # 誤差軽減のための繰り返し数

# ファイルリスト
file_list = [f"m{i}.csv" for i in range(2, 11)]

# 結果を保存するリスト
average_medians = []

for data_name in file_list:
    print(f"<<<{data_name}を実行>>>")

    # データの読み込み
    raw_data = pd.read_csv(data_name, header=None)
    data = raw_data.values  # データをnumpy配列に変換

    data_yoko = data.shape[1] if len(data.shape) > 1 else 1  # 列数
    data_tate = data.shape[0]  # 行数

    if data_yoko == 1:
        data = np.column_stack((data, np.zeros((data.shape[0], 3))))  # 列数を増やす

    step = 2              # 近接ベクトルのステップ数

    t1 = time.time()

    # ベクトル生成
    vec = [[0 for _ in range(data_yoko)] for _ in range(data_tate)]
    vector = [[0 for _ in range(data_yoko * 2)] for _ in range(data_tate)]

    for p in range(data_tate - 1):
        vec[p] = [
            data[p + 1][i] - data[p][i] for i in range(data_yoko)
        ]

    for q in range(data_tate - 1):
        vector[q] = (
            [data[q][i] for i in range(data_yoko)] +
            [data[q + 1][i] - data[q][i] for i in range(data_yoko)]
        )

    # 経験値のための繰り返し
    median_list = []
    for A in range(A_repeat):
        print(f"第{A + 1}回目")

        e_trans_list = [0 for _ in range(N_0_repeat)]

        for N in range(N_0_repeat):
            reference = random.randint(0, data_tate - step - 1)  # 基準ベクトル
            kyori = []

            for i in range(data_tate - 2):
                dis = sum([
                    pow(vector[reference][j] - vector[i][j], 2) for j in range(data_yoko)
                ]) ** 0.5
                kyori.append(dis)

            # 近接ベクトル不足時のスキップ処理
            if len(kyori) < kinbou + 1:
                print(f"Insufficient neighbors for reference {reference}. Skipping iteration.")
                continue

            syoujun = sorted(kyori)  # 距離を昇順に変換

            x = [[0 for _ in range(data_yoko)] for _ in range(kinbou + 1)]  # 基準ベクトル
            for i in range(kinbou + 1):
                x[i] = vector[kyori.index(syoujun[i])]

            y = [[0 for _ in range(data_yoko)] for _ in range(kinbou + 1)]  # T ステップ後
            for i in range(kinbou + 1):
                y[i] = vector[kyori.index(syoujun[i]) + step]

            v = [[0 for _ in range(data_yoko)] for _ in range(kinbou + 1)]
            for i in range(kinbou + 1):
                for j in range(data_yoko):
                    v[i][j] = y[i][j] - x[i][j]

            v_ave = np.mean(v, axis=0)  # vの平均値

            sigma = 0
            for i in range(kinbou + 1):
                trans_dis = sum([
                    pow(v[i][j] - v_ave[j], 2) for j in range(data_yoko)
                ])

                bunshi = trans_dis
                bunbo = sum([
                    pow(v_ave[j], 2) for j in range(data_yoko)
                ])

                if bunbo == 0:
                    sigma += 0
                else:
                    sigma += bunshi / bunbo

            e_trans = sigma / (kinbou + 1)
            e_trans_list[N] = e_trans

        e_trans_median = np.median(e_trans_list)  # 中央値
        median_list.append(e_trans_median)
        print(f"中央値: {e_trans_median}")

    # 中央値の平均を計算して保存
    average_median = sum(median_list) / len(median_list)
    average_medians.append(average_median)
    print(f"{data_name} の中央値の平均: {average_median}")

# 折れ線グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), average_medians, marker='o', linestyle='-', alpha=0.8)
plt.xlabel('m value')
plt.ylabel('Average Median')
plt.title('Average Median Values for Different m')
plt.xticks(range(2, 11))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 出力
print("\u5b8c\u4e86")
t2 = time.time()
t3 = t2 - t1
print(time.strftime("%H:%M:%S", time.gmtime(t3)))
