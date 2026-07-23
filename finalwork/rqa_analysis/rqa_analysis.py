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
    print("Usage: python rqa_analysis.py <file_name>")
    sys.exit(1)

file_name = sys.argv[2]
input_file = os.path.join(file_name)

if not os.path.exists(input_file):
    print("Can't find the file. Error!!:{input_file}")
    sys.exit(1)

#出力フォルダ
output_folder = './rqa_radius_sweep'
os.makedirs(output_folder, exist_ok=True)



def calculate_autocorrelation(series, max_lag):
    autocorr_values = []
    series = np.asarray(series, dtype=float)
    series_mean = np.mean(series)
    denominator = np.sum((series - series_mean) ** 2)
    for lag in range(1, max_lag + 1):
        if lag >= len(series):
            break
        num = np.sum((series[:-lag] - series_mean) * (series[lag:] - series_mean))
        autocorr = num / denominator if denominator != 0 else 0.0
        autocorr_values.append(autocorr)
    return autocorr_values

def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threshold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1
    return max_lag

#data name & data input
name = os.path.splitext(os.path.basename(input_file))[0]
data_points= np.loadtxt(input_file, usecols=[0], delimiter=',')


#データの読み込み・正規化
#data_points = (data_points - np.mean(data_points)) / np.std(data_points)
data_points = (data_points - np.min(data_points)) / (np.max(data_points) - np.min(data_points))

tau = determine_tau(data_points)
print(f"Delay time τ: {tau}")
        
#各radiusでRQAを実行し、RRを取得
radius_list = np.linspace(0.01, 1.0, 50) #0.01~1.0までの50点
#radius_list = np.linspace(1e-8, 5e-7, 50)
rr_list = []
for radius in radius_list:
    time_series = TimeSeries(data_points,
                             embedding_dimension=4,
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
    print(f"Number of data: {len(data_points)}, Minimum: {(4-1)*tau}")


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
