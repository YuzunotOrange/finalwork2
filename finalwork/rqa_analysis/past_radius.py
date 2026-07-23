import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


#コマンドライン引数からファイルを取得
if len(sys.argv) < 3:
    print("Usage: python radius.py <folder_path> <file_name>")
    sys.exit(1)

folder_path = sys.argv[1]
file_name = sys.argv[2]
input_file = os.path.join(folder_path, file_name)

if not os.path.exists(input_file):
    print("Can't find the file. Error!!:{input_file}")
    sys.exit(1)

#出力フォルダ
output_folder = './rqa_radius_sweep'
os.makedirs(output_folder, exist_ok=True)


# Estimation of time delay
def mutual_information(X, Y, bins=30):
    # https://qiita.com/hyt-sasaki/items/ffaab049e46f800f7cbf
    # 同時確率分布p(x,y)の計算
    p_xy, xedges, yedges = np.histogram2d(X, Y, bins=bins, density=True)

    # p(x)p(y)の計算
    p_x, _ = np.histogram(X, bins=xedges, density=True)
    p_y, _ = np.histogram(Y, bins=yedges, density=True)
    p_x_y = p_x[:, np.newaxis] * p_y

    # dx と dy
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # 積分の要素
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.where(p_xy > 0, np.log(p_xy / p_x_y), 0)
    elem = p_xy * log_term

    # 相互情報量とp(x, y), p(x)p(y)を出力
    return np.sum(elem * dx * dy), p_xy, p_x_y


#data name & data input
name = os.path.splitext(os.path.basename(input_file))[0]
data_points= np.loadtxt(input_file, usecols=[0], delimiter=',')


#データの読み込み・正規化
#data_points = (data_points - np.mean(data_points)) / np.std(data_points)
data_points = (data_points - np.min(data_points)) / (np.max(data_points) - np.min(data_points))



X=data_points
mi=np.zeros(100)
for a in range(100):
    Y=np.roll(X,a)
    mi[a], p_xy, p_x_y = mutual_information(X, Y, bins=30)
tau = int(np.argmin(mi)) #time delay
print(f"時間遅れ τ: {tau}")
        
#各radiusでRQAを実行し、RRを取得
radius_list = np.linspace(0.01, 1.0, 50) #0.01~1.0までの50点
#radius_list = np.linspace(1e-8, 5e-7, 50)
rr_list = []
for radius in radius_list:
    time_series = TimeSeries(data_points,
                             embedding_dimension=10,
                             time_delay=tau)
    settings = Settings(time_series,
                         analysis_type=Classic,
                         neighbourhood=FixedRadius(radius),
                         similarity_measure=EuclideanMetric,
                         theiler_corrector=1)
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    rr_list.append(result.recurrence_rate)
    print(f"radius: {radius:.3f}, RR {result.recurrence_rate:.5f}")
    print(f"Number of data: {len(data_points)}, Minimum: {(10-1)*tau}")


print("First tenth data:", data_points[:10])    

# Output
plt.figure()
plt.plot(radius_list, rr_list, marker='o')
plt.xlabel('Radius')
plt.ylabel('Recurrence Rate (RR)')
plt.title(f'Recurrence Rate vs Radius: {name}')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_folder, f'{name}_rr_vs_radius.png')
plt.savefig(plot_path)
plt.close()

result_path = os.path.join(output_folder, f'{name}_rr_vs_radius.csv')
np.savetxt(result_path, np.column_stack((radius_list, rr_list)), delimiter=',', header='radius,recurence_rate', comments ='')

print(f"保存完了: {plot_path}, {result_path}")
