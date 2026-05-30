import os
import sys
import numpy as np
import pandas as pd
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation

def run_rqa(data, embedding_dimension=2, time_delay=2, radius=0.65):
    time_series = TimeSeries(data,
                              embedding_dimension=embedding_dimension,
                              time_delay=time_delay)
    
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    # ライン長の最小設定
    settings.min_diagonal_line_length = 2
    settings.min_vertical_line_length = 2
    settings.white_vertical_line_length = 2

    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    
    return {
        "recurrence_rate": result.recurrence_rate,
        "determinism": result.determinism,
        "entropy": result.entropy_diagonal_lines,
        "laminarity": result.laminarity,
        "trapping_time": result.trapping_time
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_pyrqa.py <csv_directory>")
        sys.exit(1)

    folder_path = sys.argv[1]

    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            try:
                data = np.loadtxt(filepath)
                metrics = run_rqa(data)
                metrics["filename"] = filename
                results.append(metrics)
                print(f"[OK] {filename} done.")
            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")

    # 結果をCSVで保存
    df = pd.DataFrame(results)
    output_csv = os.path.join(folder_path, "rqa_batch_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ All results saved to: {output_csv}")

if __name__ == "__main__":
    main()
