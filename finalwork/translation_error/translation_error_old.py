import numpy as np
import pandas as pd
import nolds
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute correlation dimension from space-separated time series file")
    parser.add_argument("csv_file", help="Path to the space-separated file")
    args = parser.parse_args()

    # 一列目だけを読み込む
    try:
        df = pd.read_csv(args.csv_file, header=None, delim_whitespace=True)
        time_series = df.iloc[:, 0].dropna().values
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Loaded {len(time_series)} points in total.")

    if len(time_series) < 10:
        print("Warning: very short time series. Results may not be meaningful.")

    # 埋め込み次元を1〜10で試す
    embedding_dims = range(1, 11)
    corr_dims = []

    for dim in embedding_dims:
        if len(time_series) < dim:
            print(f"Skipping dim={dim}: not enough data points")
            corr_dims.append(np.nan)
            continue

        try:
            cd = nolds.corr_dim(time_series, emb_dim=dim)
            print(f"dim={dim}: correlation dimension = {cd}")
        except Exception as e:
            print(f"Error at dim={dim}: {e}")
            cd = np.nan
        corr_dims.append(cd)

    # プロット
    plt.figure(figsize=(8, 6))
    plt.plot(embedding_dims, corr_dims, 'o-', label='Correlation Dimension')
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Correlation Dimension")
    plt.yscale('log')
    plt.title(f"Correlation Dimension from {args.csv_file}")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
