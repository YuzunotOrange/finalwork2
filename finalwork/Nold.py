import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
import nolds
import shutil

def compute_correlation_dimension(series, max_dim=10):
    dims = list(range(1, max_dim + 1))
    results = []

    for dim in dims:
        if len(series) < dim * 10:
            # 適当に閾値を設定（経験的に最低必要点数を確保するため）
            print(f"Skipping dim={dim}: not enough data points")
            results.append(np.nan)
            continue
        try:
            cd = nolds.corr_dim(series, emb_dim=dim)
            print(f"dim={dim}: correlation dimension = {cd}")
            results.append(cd)
        except Exception as e:
            print(f"Error at dim={dim}: {e}")
            results.append(np.nan)
    return dims, results

def main():
    parser = argparse.ArgumentParser(description="Compare correlation dimension for new_li CSVs vs reference CSV")
    parser.add_argument("reference_csv", help="Path to lorenz.csv (or other reference)")
    parser.add_argument("--input_dir", default="new_li", help="Directory containing CSVs to compare")
    parser.add_argument("--output_dir", default="finalresult_nold", help="Directory to save result plots")
    args = parser.parse_args()

    # Clean/create output directory
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    print(f"[INFO] Created output directory: {args.output_dir}")

    # --- Load reference (Method 2) ---
    print("\n==== Computing reference (Method 2) ====")
    try:
        ref_data = pd.read_csv(args.reference_csv, header=None, delim_whitespace=True)
        ref_series = ref_data.iloc[:, 0].dropna().values
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return

    m_ref, cd_ref = compute_correlation_dimension(ref_series)
    print(f"[INFO] Computed correlation dimension for {args.reference_csv}")

    # --- Process all CSVs in input_dir (Method 1) ---
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in {args.input_dir}")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"\n==== Processing {filename} ====")
        try:
            df = pd.read_csv(filepath, header=None, delim_whitespace=True)
            series = df.iloc[:, 0].dropna().values
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        m_target, cd_target = compute_correlation_dimension(series)
        print(f"[INFO] Computed correlation dimension for {filename}")

        # --- Plot comparison ---
        plt.figure(figsize=(10, 6))
        plt.plot(m_target, cd_target, 'o-', label=f"Method 1 ({filename})")
        plt.plot(m_ref, cd_ref, 's--', label=f"Method 2 ({os.path.basename(args.reference_csv)})")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Correlation Dimension")
        plt.yscale("log")
        plt.title(f"Correlation Dimension Comparison\n{filename} vs {os.path.basename(args.reference_csv)}")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xticks(range(1, 11))
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_comparison.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[SAVED] {out_path}")

    print("\n[DONE] All comparisons saved.")

if __name__ == "__main__":
    main()
