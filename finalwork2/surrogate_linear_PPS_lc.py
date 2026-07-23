"""
PPSサロゲートデータ法による非線形性検定（欠損・補間なし）

処理内容
1. CSVの第1列を時系列として読み込む
2. 平均相互情報量（AMI）から遅れ時間 tau を推定
3. 伊藤法E1の飽和から埋め込み次元 m を推定
4. 佐野・沢田法に基づく最大リアプノフ指数を計算
5. PPSサロゲートを39本生成して、Zスコアとモンテカルロp値を計算
6. 同じ解析を5回実行して、CSVとヒストグラムを保存

実行例
    python pps_surrogate_executable.py data1.csv data2.csv

引数を省略した場合は、実行時にCSVファイル名を入力します。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors


# ==============================
# 解析設定
# ==============================
RESULT_DIR = Path("result_linear_PPS_lc")
NUM_RUNS = 5
NUM_SURROGATES = 39
MAX_LAG = 100
MAX_DIM = 10
E1_TOLERANCE = 0.05
RANDOM_SEED = 20260604


def load_csv_timeseries(csv_path: str | Path) -> np.ndarray:
    """CSVの第1列を浮動小数点時系列として読み込む。"""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    if df.empty or df.shape[1] < 1:
        raise ValueError(f"CSVにデータがありません: {csv_path}")

    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    series = series[np.isfinite(series)]

    if len(series) < 50:
        raise ValueError(
            f"有効なデータ点が少なすぎます: {csv_path} "
            f"(有効点数={len(series)}, 50点以上を推奨)"
        )

    return series


def average_mutual_information(
    x: np.ndarray,
    max_lag: int = 100,
    bins: int = 32,
) -> np.ndarray:
    """各ラグに対する平均相互情報量を計算する。"""
    x = np.asarray(x, dtype=float)
    max_lag = min(max_lag, max(1, len(x) // 4))

    ami: list[float] = []

    for lag in range(1, max_lag + 1):
        x1 = x[:-lag]
        x2 = x[lag:]

        edges = np.histogram_bin_edges(x, bins=bins)
        x1_bin = np.digitize(x1, edges[1:-1])
        x2_bin = np.digitize(x2, edges[1:-1])

        ami.append(mutual_info_score(x1_bin, x2_bin))

    return np.asarray(ami, dtype=float)


def determine_tau(series: np.ndarray, max_lag: int = 100) -> int:
    """AMIの最初の極小値をtauとする。極小値がなければ最小値を使う。"""
    ami = average_mutual_information(series, max_lag=max_lag)

    if len(ami) == 0:
        return 1

    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return i + 1

    return int(np.argmin(ami) + 1)


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens埋め込み行列を作る。"""
    x = np.asarray(x, dtype=float)
    n_vectors = len(x) - (m - 1) * tau

    if n_vectors <= 0:
        return np.empty((0, m), dtype=float)

    indices = (
        np.arange(n_vectors)[:, None]
        + tau * np.arange(m)[None, :]
    )
    return x[indices]


def itoh_e1(
    x: np.ndarray,
    max_dim: int = 10,
    tau: int = 5,
    s: int | None = None,
    k: int = 10,
    theiler: int = 0,
) -> np.ndarray:
    """伊藤法E1を計算する。"""
    x = np.asarray(x, dtype=float)

    if s is None:
        s = tau

    e1_values: list[float] = []

    for m in range(1, max_dim + 1):
        xm = embed(x, m, tau)
        total = len(xm)

        if total <= s:
            break

        valid_count = total - s
        x_now = xm[:valid_count]
        k_eff = min(k, valid_count - 1)

        if k_eff < 1:
            break

        n_query = min(k_eff + 21, valid_count)
        nn = NearestNeighbors(
            n_neighbors=n_query,
            algorithm="kd_tree",
        ).fit(x_now)

        distances, indices = nn.kneighbors(x_now)
        ratios: list[float] = []

        for i in range(valid_count):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]

            if theiler > 0:
                valid = np.abs(neighbor_indices - i) > theiler
                neighbor_indices = neighbor_indices[valid]
                neighbor_distances = neighbor_distances[valid]

            valid = (neighbor_indices + s) < total
            neighbor_indices = neighbor_indices[valid]
            neighbor_distances = neighbor_distances[valid]

            if len(neighbor_indices) == 0:
                continue

            use = min(k_eff, len(neighbor_indices))
            selected = neighbor_indices[:use]
            initial_distance = float(np.mean(neighbor_distances[:use]))

            if initial_distance < 1e-12:
                continue

            future_distance = float(
                np.mean(
                    np.linalg.norm(
                        xm[i + s] - xm[selected + s],
                        axis=1,
                    )
                )
            )
            ratios.append(future_distance / initial_distance)

        if ratios:
            e1_values.append(float(np.mean(ratios)))

    return np.asarray(e1_values, dtype=float)


def determine_embedding_dimension(
    e1_values: np.ndarray,
    tolerance: float = 0.05,
    default_m: int = 2,
) -> int:
    """隣接するE1値の比が1に近づいた最初の次元を採用する。"""
    if len(e1_values) < 2:
        return default_m

    for i in range(1, len(e1_values)):
        previous = e1_values[i - 1]

        if abs(previous) < 1e-12:
            continue

        ratio = e1_values[i] / previous

        if abs(ratio - 1.0) < tolerance:
            return i + 1

    # 飽和が見つからない場合は計算できた最大次元を採用する。
    return max(default_m, min(len(e1_values), MAX_DIM))


def sano_sawada_lyapunov(
    data: np.ndarray,
    m: int = 3,
    tau: int = 1,
    n_neighbors: int = 30,
    theiler: int | None = None,
    sample_step: int = 10,
) -> float:
    """
    局所線形写像の最大特異値から最大リアプノフ指数を推定する。
    """
    data = np.asarray(data, dtype=float)
    embedded = embed(data, m, tau)
    num_points = len(embedded)
    evolution_step = 1

    if theiler is None:
        theiler = tau * m

    if num_points <= max(n_neighbors + 2, evolution_step + 1):
        return float("nan")

    local_exponents: list[float] = []
    search_count = num_points - evolution_step

    for i in range(0, search_count, sample_step):
        diff = embedded[:search_count] - embedded[i]
        distances = np.linalg.norm(diff, axis=1)

        start = max(0, i - theiler)
        end = min(search_count, i + theiler + 1)
        distances[start:end] = np.inf

        finite_indices = np.where(np.isfinite(distances))[0]

        if len(finite_indices) < m + 1:
            continue

        neighbor_count = min(n_neighbors, len(finite_indices))
        nearest = finite_indices[
            np.argpartition(
                distances[finite_indices],
                neighbor_count - 1,
            )[:neighbor_count]
        ]

        z_now = embedded[nearest] - embedded[i]
        z_next = (
            embedded[nearest + evolution_step]
            - embedded[i + evolution_step]
        )

        try:
            # z_now @ A ≈ z_next
            jacobian, _, rank, _ = np.linalg.lstsq(
                z_now,
                z_next,
                rcond=None,
            )

            if rank < 1:
                continue

            singular_values = np.linalg.svd(
                jacobian,
                compute_uv=False,
            )

            if len(singular_values) > 0 and singular_values[0] > 0:
                local_exponents.append(
                    float(np.log(singular_values[0]))
                )
        except np.linalg.LinAlgError:
            continue

    if not local_exponents:
        return float("nan")

    return float(np.mean(local_exponents))


def pps_surrogate(
    x: np.ndarray,
    m: int,
    tau: int,
    rng: np.random.Generator,
    radius: float | None = None,
) -> np.ndarray:
    """PPS（pseudo-periodic surrogate）を1本生成する。"""
    x = np.asarray(x, dtype=float)
    embedded = embed(x, m, tau)
    vector_count = len(embedded)

    if vector_count < 20:
        return x.copy()

    if radius is None:
        sample_count = min(1000, vector_count)
        sample_indices = rng.choice(
            vector_count,
            size=sample_count,
            replace=False,
        )
        sample = embedded[sample_indices]

        pairwise_distances = np.linalg.norm(
            sample[:, None, :] - sample[None, :, :],
            axis=2,
        )
        positive_distances = pairwise_distances[pairwise_distances > 0]

        if len(positive_distances) == 0:
            radius = 1.0
        else:
            radius = float(np.percentile(positive_distances, 5))

    radius = max(float(radius), 1e-12)

    current_index = int(rng.integers(vector_count))
    surrogate_indices = [current_index]
    theiler = tau * m
    all_indices = np.arange(vector_count)

    for _ in range(vector_count - 1):
        distances = np.linalg.norm(
            embedded - embedded[current_index],
            axis=1,
        )
        probabilities = np.exp(-distances / radius)

        probabilities[
            np.abs(all_indices - current_index) <= theiler
        ] = 0.0
        probabilities[current_index] = 0.0

        probability_sum = float(probabilities.sum())

        if not np.isfinite(probability_sum) or probability_sum <= 0:
            next_base = int(rng.integers(vector_count))
        else:
            probabilities /= probability_sum
            next_base = int(
                rng.choice(vector_count, p=probabilities)
            )

        current_index = min(next_base + 1, vector_count - 1)
        surrogate_indices.append(current_index)

    return x[np.asarray(surrogate_indices, dtype=int)]


def surrogate_z_test(
    original_value: float,
    surrogate_values: np.ndarray,
) -> dict[str, float]:
    """サロゲート分布に対するZスコアを返す。"""
    surrogate_values = np.asarray(surrogate_values, dtype=float)
    surrogate_values = surrogate_values[np.isfinite(surrogate_values)]

    if not np.isfinite(original_value) or len(surrogate_values) < 2:
        return {
            "z_score": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    mean = float(np.mean(surrogate_values))
    std = float(np.std(surrogate_values, ddof=1))

    if std <= 0 or not np.isfinite(std):
        z_score = float("nan")
    else:
        z_score = float((original_value - mean) / std)

    return {
        "z_score": z_score,
        "mean": mean,
        "std": std,
    }


def significance_label(z_score: float) -> str:
    """両側Z検定の有意水準ラベル。"""
    if not np.isfinite(z_score):
        return "NaN"
    if abs(z_score) > 2.58:
        return "1%"
    if abs(z_score) > 1.96:
        return "5%"
    return "NS"


def monte_carlo_p_value(
    original_value: float,
    surrogate_values: np.ndarray,
) -> float:
    """
    元データがサロゲート分布の両端のどちらに外れても検出する
    両側モンテカルロp値。
    """
    surrogate_values = np.asarray(surrogate_values, dtype=float)
    surrogate_values = surrogate_values[np.isfinite(surrogate_values)]

    if not np.isfinite(original_value) or len(surrogate_values) == 0:
        return float("nan")

    center = float(np.median(surrogate_values))
    original_distance = abs(original_value - center)
    surrogate_distances = np.abs(surrogate_values - center)

    extreme_count = int(
        np.sum(surrogate_distances >= original_distance)
    )

    return float(
        (extreme_count + 1) / (len(surrogate_values) + 1)
    )


def safe_name(name: str) -> str:
    """保存フォルダ名として使える簡易名称へ変換する。"""
    result = name.replace(" ", "_")
    for character in '()[]{}<>:"/\\|?*':
        result = result.replace(character, "")
    return result or "dataset"


def analyze_dataset(
    name: str,
    base_data: np.ndarray,
    result_dir: Path,
    num_runs: int,
    num_surrogates: int,
    base_seed: int,
) -> list[dict[str, object]]:
    """1つの時系列を複数回解析する。"""
    dataset_results: list[dict[str, object]] = []

    tau_est = determine_tau(base_data, max_lag=MAX_LAG)
    tau_est = int(np.clip(tau_est, 1, 50))

    e1_values = itoh_e1(
        base_data,
        max_dim=MAX_DIM,
        tau=tau_est,
        theiler=tau_est,
    )
    m_est = determine_embedding_dimension(
        e1_values,
        tolerance=E1_TOLERANCE,
    )

    print(f"\nData: {name}")
    print(f"Length: {len(base_data)}")
    print(f"Estimated tau: {tau_est}")
    print(f"Estimated m: {m_est}")
    print(f"E1 values: {e1_values}")

    system_dir = result_dir / safe_name(name)
    system_dir.mkdir(parents=True, exist_ok=True)

    parameter_df = pd.DataFrame(
        {
            "parameter": ["length", "tau", "m"],
            "value": [len(base_data), tau_est, m_est],
        }
    )
    parameter_df.to_csv(
        system_dir / "estimated_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for run in range(1, num_runs + 1):
        print("\n================================")
        print(f"Data : {name}")
        print(f"Run  : {run}/{num_runs}")
        print("================================")

        run_dir = system_dir / f"run_{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(base_seed + run)

        pd.DataFrame(
            {
                "time": np.arange(len(base_data)),
                "original": base_data,
            }
        ).to_csv(
            run_dir / f"timeseries_{safe_name(name)}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Calculating original Lyapunov exponent...")
        original_lambda = sano_sawada_lyapunov(
            base_data,
            m=m_est,
            tau=tau_est,
            theiler=tau_est * m_est,
        )

        surrogate_lambdas: list[float] = []

        print("Calculating PPS surrogates...")
        for surrogate_number in range(1, num_surrogates + 1):
            print(
                f"Run {run}: surrogate "
                f"{surrogate_number}/{num_surrogates}"
            )

            surrogate = pps_surrogate(
                base_data,
                m=m_est,
                tau=tau_est,
                rng=rng,
            )
            surrogate_lambda = sano_sawada_lyapunov(
                surrogate,
                m=m_est,
                tau=tau_est,
                theiler=tau_est * m_est,
            )
            surrogate_lambdas.append(surrogate_lambda)

        surrogate_array = np.asarray(
            surrogate_lambdas,
            dtype=float,
        )
        finite_surrogates = surrogate_array[
            np.isfinite(surrogate_array)
        ]

        z_result = surrogate_z_test(
            original_lambda,
            finite_surrogates,
        )
        p_value = monte_carlo_p_value(
            original_lambda,
            finite_surrogates,
        )

        result_df = pd.DataFrame(
            {
                "type": (
                    ["original"]
                    + ["PPS_surrogate"] * len(surrogate_array)
                ),
                "lambda": (
                    [original_lambda]
                    + surrogate_array.tolist()
                ),
            }
        )
        result_df.to_csv(
            run_dir / f"lyapunov_results_{safe_name(name)}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        dataset_results.append(
            {
                "Data": name,
                "Run": run,
                "N": len(base_data),
                "tau": tau_est,
                "m": m_est,
                "Original lambda": original_lambda,
                "PPS Mean": z_result["mean"],
                "PPS Std": z_result["std"],
                "PPS Z-score": z_result["z_score"],
                "PPS Significance": significance_label(
                    z_result["z_score"]
                ),
                "Monte Carlo p-value": p_value,
                "Valid Surrogates": len(finite_surrogates),
            }
        )

        if len(finite_surrogates) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(
                finite_surrogates,
                bins=15,
                alpha=0.7,
                edgecolor="black",
                label="PPS surrogates",
            )
            plt.axvline(
                original_lambda,
                linestyle="--",
                linewidth=2,
                label=(
                    "Original "
                    rf"($\lambda$={original_lambda:.3f})"
                ),
            )
            plt.title(
                f"{name}\n"
                f"Run={run}/{num_runs}, "
                f"tau={tau_est}, m={m_est}, "
                f"p={p_value:.4f}"
            )
            plt.xlabel("Maximum Lyapunov exponent")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                run_dir / f"result_{safe_name(name)}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        print(f"Original lambda = {original_lambda}")
        print(f"PPS mean = {z_result['mean']}")
        print(f"PPS std = {z_result['std']}")
        print(f"PPS Z-score = {z_result['z_score']}")
        print(
            "Significance = "
            f"{significance_label(z_result['z_score'])}"
        )
        print(f"Monte Carlo p-value = {p_value}")

    return dataset_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CSV時系列に対してPPSサロゲート検定を実行します。"
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="解析するCSVファイル。複数指定可能。",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"解析回数（既定値: {NUM_RUNS}）",
    )
    parser.add_argument(
        "--surrogates",
        type=int,
        default=NUM_SURROGATES,
        help=f"各回のサロゲート数（既定値: {NUM_SURROGATES}）",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help=f"結果保存先（既定値: {RESULT_DIR}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"乱数シード（既定値: {RANDOM_SEED}）",
    )
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="既存の結果フォルダを削除せず追記する。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.runs < 1:
        raise ValueError("--runs は1以上にしてください。")
    if args.surrogates < 1:
        raise ValueError("--surrogates は1以上にしてください。")

    csv_files = list(args.csv_files)

    if not csv_files:
        print("解析したいCSVファイル名を入力してください。")
        print("複数の場合はカンマで区切ります。")
        print("例: data1.csv,data2.csv")
        entered = input(">> ").strip()
        csv_files = [
            item.strip()
            for item in entered.split(",")
            if item.strip()
        ]

    if not csv_files:
        raise ValueError("CSVファイルが指定されていません。")

    if args.result_dir.exists() and not args.keep_results:
        print(f"Cleaning up {args.result_dir}...")
        shutil.rmtree(args.result_dir)

    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for file_number, csv_file in enumerate(csv_files):
        csv_path = Path(csv_file)

        try:
            data = load_csv_timeseries(csv_path)
            name = csv_path.stem

            results = analyze_dataset(
                name=name,
                base_data=data,
                result_dir=args.result_dir,
                num_runs=args.runs,
                num_surrogates=args.surrogates,
                base_seed=args.seed + file_number * 10000,
            )
            all_results.extend(results)

            pd.DataFrame(all_results).to_csv(
                args.result_dir / "surrogate_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {csv_path}: {message}")
            errors.append(
                {
                    "file": str(csv_path),
                    "error": message,
                }
            )

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = args.result_dir / "surrogate_summary.csv"
        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\nSaved final summary: {summary_path}")
    else:
        print("\n解析に成功したデータはありませんでした。")

    if errors:
        error_path = args.result_dir / "errors.csv"
        pd.DataFrame(errors).to_csv(
            error_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Errors were saved to: {error_path}")


if __name__ == "__main__":
    main()