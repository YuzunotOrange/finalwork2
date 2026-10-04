import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import time
import random


N_0_REPEAT = 50
KINBOU = 4
A_REPEAT = 10
M_RANGE = range(1, 10)
MAX_LAG = 100

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_csv_timeseries(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        lower_map = {str(c).lower(): c for c in df.columns}

        for preferred in ["flux", "value"]:
            if preferred in lower_map:
                series = pd.to_numeric(
                    df[lower_map[preferred]],
                    errors="coerce"
                ).dropna().to_numpy(dtype=float)

                if len(series) > 0:
                    return series

        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        usable_cols = [
            c for c in numeric_df.columns
            if numeric_df[c].notna().sum() > 0
        ]

        if usable_cols:
            series = (
                numeric_df[usable_cols[-1]]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(series) > 0:
                return series

    except Exception:
        pass

    df = pd.read_csv(
        csv_path,
        header=None,
        sep=r"\s+|,",
        engine="python"
    )

    series = (
        pd.to_numeric(df.iloc[:, 0], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(series) == 0:
        raise ValueError(
            f"数値時系列を読み込めませんでした: {csv_path}"
        )

    return series


def calculate_autocorrelation(series, max_lag):
    autocorr_values = []
    series_mean = np.mean(series)

    denominator = np.sum(
        (series - series_mean) ** 2
    )

    for lag in range(1, max_lag + 1):

        if lag >= len(series):
            break

        numerator = np.sum(
            (series[:-lag] - series_mean)
            * (series[lag:] - series_mean)
        )

        if denominator == 0:
            autocorr = 0.0
        else:
            autocorr = numerator / denominator

        autocorr_values.append(autocorr)

    return autocorr_values


def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(
        series,
        max_lag
    )

    if len(autocorr) == 0:
        raise ValueError(
            "tauを推定するにはデータ点数が不足しています。"
        )

    threshold = 1 / np.e

    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1

    return len(autocorr)


def calculate_translation_error(series):

    series = np.asarray(
        series,
        dtype=float
    )

    series = series[
        np.isfinite(series)
    ]

    if len(series) < 10:
        raise ValueError(
            "データ点数が少なすぎます。"
        )

    std = np.std(series)

    if std == 0:
        raise ValueError(
            "標準偏差が0のため標準化できません。"
        )

    series = (
        series - np.mean(series)
    ) / std

    time_delay = determine_tau(
        series,
        MAX_LAG
    )

    average_medians = []

    for m in M_RANGE:

        print(
            f"    embedding dimension m = {m}"
        )

        n_vectors = (
            len(series)
            - m * time_delay
        )

        if n_vectors <= KINBOU + 1:
            average_medians.append(np.nan)
            print(
                "      -> データ不足のためNaN"
            )
            continue

        embedded_vectors = np.array([
            [
                series[i + j * time_delay]
                for j in range(m)
            ]
            for i in range(n_vectors)
        ])

        data_tate = embedded_vectors.shape[0]

        median_list = []

        for _ in range(A_REPEAT):

            e_trans_list = []

            for _ in range(N_0_REPEAT):

                ref = random.randint(
                    0,
                    data_tate - 2
                )

                dists = np.linalg.norm(
                    embedded_vectors
                    - embedded_vectors[ref],
                    axis=1
                )

                nearest_indices = np.array([
                    i
                    for i in np.argsort(dists)
                    if i + 1 < data_tate
                ][:KINBOU + 1])

                if len(nearest_indices) < KINBOU + 1:
                    continue

                x = embedded_vectors[
                    nearest_indices
                ]

                y = embedded_vectors[
                    nearest_indices + 1
                ]

                v = y - x

                v_ave = np.mean(
                    v,
                    axis=0
                )

                bunbo = np.sum(
                    v_ave ** 2
                )

                if bunbo == 0:

                    e_trans = 0.0

                else:

                    sigma = np.sum([
                        np.sum(
                            (vi - v_ave) ** 2
                        ) / bunbo
                        for vi in v
                    ])

                    e_trans = (
                        sigma
                        / (KINBOU + 1)
                    )

                e_trans_list.append(
                    e_trans
                )

            if e_trans_list:
                median_list.append(
                    np.median(e_trans_list)
                )

        if median_list:
            average_medians.append(
                float(
                    np.mean(median_list)
                )
            )
        else:
            average_medians.append(
                np.nan
            )

    values = np.asarray(
        average_medians,
        dtype=float
    )

    finite_mask = np.isfinite(values)

    max_change_from_m = np.nan
    max_change_to_m = np.nan
    threshold = np.nan

    if np.sum(finite_mask) >= 2:

        valid_indices = np.where(
            finite_mask
        )[0]

        valid_values = values[
            finite_mask
        ]

        diff = np.diff(
            valid_values
        )

        if len(diff) > 0:

            local_index = int(
                np.argmax(
                    np.abs(diff)
                )
            )

            max_diff_index = (
                valid_indices[local_index]
            )

            start = max(
                0,
                max_diff_index - 1
            )

            end = min(
                len(values),
                max_diff_index + 2
            )

            threshold = float(
                np.nanmean(
                    values[start:end]
                )
            )

            max_change_from_m = start + 1
            max_change_to_m = end

    return {
        "tau": time_delay,
        "translation_errors": average_medians,
        "max_change_from_m": max_change_from_m,
        "max_change_to_m": max_change_to_m,
        "threshold": threshold,
    }


def save_plot(name, result, output_path):

    m_values = list(M_RANGE)

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        m_values,
        result["translation_errors"],
        marker="o"
    )

    plt.xlabel(
        "Embedding dimension m"
    )

    plt.ylabel(
        "Translation Error"
    )

    plt.title(
        f"{name}\n"
        f"Translation Error vs Embedding Dimension "
        f"(tau={result['tau']})"
    )

    plt.xticks(m_values)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    t1 = time.time()

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    data_dir = os.path.join(
        base_dir,
        "Bhatta_20"
    )

    result_dir = os.path.join(
        base_dir,
        "result_translationE_Bhatta"
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"{data_dir} が見つかりません"
        )

    if os.path.exists(result_dir):
        print(
            f"Cleaning up {result_dir}..."
        )
        shutil.rmtree(result_dir)

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    csv_files = sorted([
        os.path.join(
            data_dir,
            f
        )
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ])

    if not csv_files:
        raise FileNotFoundError(
            f"{data_dir} にCSVファイルがありません。"
        )

    print(
        f"{len(csv_files)} 個のCSVを検出しました。"
    )

    datasets = {}

    for csv_path in csv_files:

        name = os.path.splitext(
            os.path.basename(csv_path)
        )[0]

        print(
            f"Loading : {name}"
        )

        try:
            datasets[name] = (
                load_csv_timeseries(
                    csv_path
                )
            )

        except Exception as e:
            print(
                f"[ERROR] {name} の読み込みに失敗: {e}"
            )

    all_results = []

    for name, base_data in datasets.items():

        print(
            "\n" + "=" * 70
        )

        print(
            f"Analysis : {name}"
        )

        print(
            "=" * 70
        )

        file_safe_name = (
            name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        system_dir = os.path.join(
            result_dir,
            file_safe_name
        )

        os.makedirs(
            system_dir,
            exist_ok=True
        )

        try:

            result = (
                calculate_translation_error(
                    base_data
                )
            )

            image_path = os.path.join(
                system_dir,
                f"{file_safe_name}_translation_error.png"
            )

            save_plot(
                name,
                result,
                image_path
            )

            detail_df = pd.DataFrame({
                "dataset":
                    [name] * len(list(M_RANGE)),
                "tau":
                    [result["tau"]] * len(list(M_RANGE)),
                "m":
                    list(M_RANGE),
                "translation_error":
                    result["translation_errors"],
            })

            detail_csv = os.path.join(
                system_dir,
                f"{file_safe_name}_translation_error.csv"
            )

            detail_df.to_csv(
                detail_csv,
                index=False,
                encoding="utf-8-sig"
            )

            row = {
                "dataset":
                    name,
                "n_points":
                    len(base_data),
                "tau":
                    result["tau"],
                "max_change_from_m":
                    result["max_change_from_m"],
                "max_change_to_m":
                    result["max_change_to_m"],
                "threshold":
                    result["threshold"],
            }

            for m, error in zip(
                M_RANGE,
                result["translation_errors"]
            ):

                row[
                    f"TE_m{m}"
                ] = error

            all_results.append(row)

            print(
                f"tau = {result['tau']}"
            )

            print(
                "Translation Error =",
                result["translation_errors"]
            )

            print(
                f"Saved image : {image_path}"
            )

            print(
                f"Saved CSV   : {detail_csv}"
            )

        except Exception as e:

            print(
                f"[ERROR] {name} の解析に失敗: {e}"
            )

            row = {
                "dataset": name,
                "n_points": len(base_data),
                "tau": np.nan,
                "max_change_from_m": np.nan,
                "max_change_to_m": np.nan,
                "threshold": np.nan,
                "error": str(e),
            }

            for m in M_RANGE:
                row[
                    f"TE_m{m}"
                ] = np.nan

            all_results.append(row)

    summary_df = pd.DataFrame(
        all_results
    )

    summary_csv = os.path.join(
        result_dir,
        "translation_error_all_results.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    t2 = time.time()

    print(
        "\n" + "=" * 70
    )

    print("Complete")

    print(
        f"Summary CSV : {summary_csv}"
    )

    print(
        "Processing time:",
        time.strftime(
            "%H:%M:%S",
            time.gmtime(t2 - t1)
        )
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()