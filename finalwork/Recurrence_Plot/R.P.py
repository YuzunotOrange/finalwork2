from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation
import numpy as np
import matplotlib.pyplot as plt

# 例：データ読み込み（1列目）
x = np.loadtxt("lorenz.csv", usecols=[0])

# 埋め込みせずに使う（固定）：m=1, τ=1
ts = TimeSeries(x, embedding_dimension=1, time_delay=1)

# 閾値（radius）は固定でもよいし、データのスケールから決めてもOK
# 例1: 固定半径
# radius = 0.5

# 例2: 標準偏差ベース
radius = 0.2 * np.std(x)

settings = Settings(
    ts,
    analysis_type=Classic,               # 環境により Classic() が必要なら変更
    neighbourhood=FixedRadius(radius),
    similarity_measure=EuclideanMetric   # 同上：EuclideanMetric()
)

rp = RPComputation.create(settings).run()
M = np.array(rp.recurrence_matrix)

plt.imshow(M, cmap="binary", origin="lower", interpolation="none")
plt.title("Recurrence Plot (m=1, tau=1)")
plt.xlabel("Time"); plt.ylabel("Time")
plt.show()
