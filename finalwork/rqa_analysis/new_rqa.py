from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

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
    elem = p_xy * np.ma.log(p_xy / p_x_y)
    # 相互情報量とp(x, y), p(x)p(y)を出力
    return np.sum(elem * dx * dy), p_xy, p_x_y

#引数からディレクトリを取得
input_folder = './new_li'
csv_files = sorted(glob.glob(os.path.join(input_folder, '*_liner.csv')))

if not csv_files:
    print(f"No _liner.csv files found in folder: {input_folder}")
    sys.exit(1)

#全ファイル処理ループ
for fname in csv_files:
    try:
    
        #diameter =sys.argv[2]
        name = os.path.splitext(os.path.basename(fname))[0] 
        # name = 'random'
        #radius = float(diameter)*0.1/2.0
        #print(radius)


        # data input
        data_points= np.loadtxt(fname,usecols=[0], delimiter=',')
        #data_points= np.random.normal(0.0,1.0,1024)
        #print(data_points)

        # 振幅の10%とする
        amplitude_differences = np.abs(np.diff(data_points))
        max_amplitude = np.max(amplitude_differences)
        radius = 0.1 * max_amplitude
        print(f"Amplitude: {max_amplitude:.5f}, Radius: {radius:.5f}")

        X=data_points
        mi=np.zeros(100)
        for a in range(0,100):
            Y=np.roll(X,a)
            mi[a], p_xy, p_x_y = mutual_information(X, Y, bins=30)
        tau = int(np.argmin(mi)) #time delay
        print(f"時間遅れ τ: {tau}")
        
        embedding_dimension = 10
        time_delay = tau
        time_series = TimeSeries(data_points,
                             embedding_dimension=10,
                             time_delay=6)
        settings = Settings(time_series,
                         analysis_type=Classic,
                         neighbourhood=FixedRadius(radius),
                         similarity_measure=EuclideanMetric,
                         theiler_corrector=1)
        computation = RQAComputation.create(settings, verbose=True)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_length = 2

     # Output
        imagename = name +'_'+str(tau)+'_rp-sigma.png'
        outname = name + '_'+str(tau)+'_rqa-sigma.txt'
        print(result)
        with open(outname,'w') as f:
            print(result, file = f)

        computation = RPComputation.create(settings)
        result = computation.run()
        ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,imagename)
    except Exception as e:
        print(f"Error processing file {fname}: {e}")