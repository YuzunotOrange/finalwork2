import math
import matplotlib.pyplot as plt

MAX_HARMONICS = 300   # フーリエ展開の項数
SAMPLES = 2048        # サンプル数（多いほど滑らか）
PERIODS = 5        # 表示する周期数
PI = math.pi

x = []
y = []

for i in range(SAMPLES + 1):
    t = i / SAMPLES * PERIODS  # t: [0, PERIODS]
    t_mod = t % 1              # フーリエ展開の基準は1周期なので、modで周期化
    out = 0.5                  # 定数項 a_0 = 1/2
    for k in range(1, MAX_HARMONICS + 1):
        out += (-1 / (PI * k)) * math.sin(2 * PI * k * t_mod)
    x.append(t)
    y.append(out)

# プロット
plt.figure(figsize=(12, 4))
plt.plot(x, y, label='Sawtooth wave f(t) = t % 1', color='blue')
plt.title('Periodic Sawtooth Wave via Fourier Series')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
