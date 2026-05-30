import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# フォントの設定（例: メイリオ）
font_path = 'C:/Windows/Fonts/meiryo.ttc'
font_prop = fm.FontProperties(fname=font_path)

# 定数
Q = 1  # 電荷の大きさ（任意の単位）
epsilon_0 = 8.854e-12  # 真空の誘電率（F/m）
R = 1  # 半径の単位

# rの範囲
r = np.linspace(0, 10*R, 500)

# 電場の計算
E = np.piecewise(r, [r < R, (r >= R) & (r <= 2*R), r > 2*R],
                 [0, 0, lambda r: (1/(4 * np.pi * epsilon_0)) * (Q / r**2)])

# グラフの作成
plt.figure(figsize=(10, 6))
plt.plot(r, E, label='$E(r)$')
plt.axvline(R, color='k', linestyle='--', label='$r = R$')
plt.axvline(2*R, color='k', linestyle='--', label='$r = 2R$')
plt.axhline(0, color='k', linewidth=0.5)

# グラフの装飾
plt.title('電場の大きさ $E$ 距離 $r$　の関係', fontproperties=font_prop)
plt.xlabel('距離 $r$ (単位: R)', fontproperties=font_prop)
plt.ylabel('電場の大きさ $E$', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.grid(True)
plt.ylim(bottom=-0.1)
plt.xlim(0, 10*R)

# 結果の表示
plt.show()
