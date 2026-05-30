import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォントプロパティの設定
font_path = 'C:/Windows/Fonts/meiryo.ttc'
font_prop = fm.FontProperties(fname=font_path)

# 定数
Q = 1  # 電荷
epsilon_0 = 8.854187817e-12  # 真空の誘電率
R = 1  # 導体球の半径

# r の範囲
r = np.linspace(0, 10*R, 1000)
E = np.zeros_like(r)
Phi = np.zeros_like(r)

# 電場 E(r) の計算
E[r >= R] = Q / (4 * np.pi * epsilon_0 * r[r >= R]**2)

# 電位 Phi(r) の計算
Phi[r >= R] = Q / (4 * np.pi * epsilon_0 * r[r >= R])
Phi[r < R] = Q / (4 * np.pi * epsilon_0 * R)

# グラフの作成
fig, ax1 = plt.subplots()

ax1.set_xlabel('距離 r (m)', fontproperties=font_prop)
ax1.set_ylabel('電場 E (N/C)', color='tab:red', fontproperties=font_prop)
ax1.plot(r, E, color='tab:red', label='電場 E(r)')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('電位 Φ (V)', color='tab:blue', fontproperties=font_prop)
ax2.plot(r, Phi, color='tab:blue', label='電位 Φ(r)')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.tight_layout()
plt.title('導体球の電場と電位', fontproperties=font_prop)
plt.show()
