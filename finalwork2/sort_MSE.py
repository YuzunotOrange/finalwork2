from pathlib import Path
import pandas as pd

# MSEフォルダ
BASE_DIR = Path("MSE_result")

# 出力先
OUTPUT_DIR = Path("sorted_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 各力学系フォルダを探索
for system_dir in BASE_DIR.iterdir():
    if system_dir.is_dir():

        csv_path = system_dir / "pattern_mse_results.csv"

        if csv_path.exists():
            print(f"処理中: {system_dir.name}")

            # CSV読込
            df = pd.read_csv(csv_path)

            # pattern ごとにソート
            sorted_df = df.sort_values(by=["pattern", "rate"])

            # 出力先
            output_path = OUTPUT_DIR / f"{system_dir.name}_sorted.csv"

            # 保存
            sorted_df.to_csv(output_path, index=False)

            print(f"保存完了: {output_path}")
print("\n全ての力学系データの並び換えが完了しました。")