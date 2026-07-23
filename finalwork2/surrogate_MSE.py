import os
import glob
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

base_dir = "./result_linear_defect"
output_root_dir = "./MSE_result"

if os.path.exists(output_root_dir):
    shutil.rmtree(output_root_dir)

os.makedirs(output_root_dir, exist_ok=True)

search_pattern = os.path.join(base_dir, "*", "*", "*", "*.csv")
files = glob.glob(search_pattern)

print(f"Found {len(files)} CSV files.\n")

results = []

for file_path in files:

    path_parts = file_path.split(os.sep)

    file_name = path_parts[-1]
    pattern = path_parts[-2]
    rate = path_parts[-3]
    ts_type = path_parts[-4]

    try:
        df = pd.read_csv(file_path)

        df.columns = df.columns.str.strip()

        if (
            'original' in df.columns and
            'interpolated' in df.columns and
            'missing' in df.columns
        ):

            # 欠損部分のみ評価
            missing_mask = df["missing"].isna()

            true_values = df.loc[missing_mask, "original"]
            interp_values = df.loc[missing_mask, "interpolated"]

            mse = mean_squared_error(true_values, interp_values)

            results.append({
                "system_type": ts_type,
                "rate": rate,
                "pattern": pattern,
                "MSE": mse
            })

        else:
            print(f"[WARNING] Missing columns: {file_path}")

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")

if results:

    df_all = pd.DataFrame(results)

    unique_systems = df_all["system_type"].unique()

    for system in unique_systems:

        df_sub = df_all[df_all["system_type"] == system].copy()

        df_sub = df_sub.sort_values(
            by=["rate", "pattern"]
        ).reset_index(drop=True)

        system_folder_path = os.path.join(
            output_root_dir,
            system
        )

        os.makedirs(system_folder_path, exist_ok=True)

        output_file_path = os.path.join(
            system_folder_path,
            "pattern_mse_results.csv"
        )

        df_output = df_sub[
            ["rate", "pattern", "MSE"]
        ]

        df_output.to_csv(
            output_file_path,
            index=False,
            encoding="utf-8"
        )

        print(f"[SAVED] {output_file_path}")

else:
    print("[WARNING] No valid data found.")