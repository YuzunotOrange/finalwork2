
# rqa_filename_only.py
# 使い方:
#   python rqa_filename_only.py your_data.csv
#   python rqa_filename_only.py lorenz.csv           # τ=5 固定
# オプション:
#   --save_rp          RP (再帰率≈5%) も保存
#   --eps_min/--eps_max/--eps_steps/--lmin  など微調整

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # itho_e1 で使用
# 既存 import 群の直後あたりに追加
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    from pyrqa.embedding import Embedding
    from pyrqa.analysis_type import Classic
    _HAS_PYRQA = True
except Exception:
    _HAS_PYRQA = False


# ========= 設定（デフォルト） =========
MAX_DIM = 10          # ★ 最大埋め込み次元は 10 に固定
K_FRAC  = 0.50        # E1 推定で使う近傍比（系列の50%）
MAX_LAG_TAU = 100     # τ 推定の上限ラグ（lorenz.csvは無関係：固定5）

# ========= （変更しない）既存5関数 =========
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

def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if s is None: s = tau
    if k is None: k = max(1, int(0.05 * N))

    E1 = []
    for m in range(1, max_dim + 1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0:
            break

        valid_M = M - s
        X_now = Xm[:valid_M]

        k_eff = min(k, valid_M - 1)
        if k_eff < 1:
            break
        over_k = min(valid_M - 1, k_eff + 20)

        nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm='auto', metric="euclidean")
        nn.fit(X_now)
        dists, indxs = nn.kneighbors(X_now)  # include itself

        ratios = []
        for i in range(valid_M):
            cand_j = indxs[i, 1:]  # 自身以外
            cand_d = dists[i, 1:]

            # Theiler window
            if theiler > 0:
                mask = np.abs(cand_j - i) > theiler
                cand_j = cand_j[mask]
                cand_d = cand_d[mask]

            if cand_j.size == 0:
                continue

            # 未来が有効 (i+s, j+s < M)
            mask2 = (cand_j + s) < M
            cand_j = cand_j[mask2]
            cand_d = cand_d[mask2]
            if cand_j.size == 0:
                continue

            # 上位k
            use = min(k_eff, cand_j.size)
            jj = cand_j[:use]
            d_now_i = cand_d[:use].mean()
            if d_now_i < 1e-12:
                continue

            # s ステップ後の距離
            d_fut_i = np.linalg.norm(Xm[i+s] - Xm[jj+s], axis=1).mean()
            ratios.append(d_fut_i / d_now_i)

        if not ratios:
            continue
        E1.append(float(np.mean(ratios)))

    return np.array(E1, dtype=float)

def estimate_min_dimension(E1, eps=0.05, win=3):
    E1 = np.asarray(E1, dtype=float)
    finite_mask = np.isfinite(E1)
    E1f = E1[finite_mask]
    if len(E1f) == 0:
        return 1
    if len(E1f) <= 1:
        return len(E1f)
    for m in range(1, len(E1) - win + 1):
        seg = E1[m:m+win]
        dif = np.abs(np.diff(seg))
        base = np.maximum(np.abs(seg[:-1]), 1e-12)
        if np.all(dif / base <= eps):
            return m + 1
    return len(E1)

# ========= RQAユーティリティ =========
def sanitize_filename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^\w\-.]+", "_", base)
    return base or "output"

def load_series_first_col_auto(path: str) -> np.ndarray:
    """
    1列目のみ解析。lorenz.csv は空白区切り、それ以外は区切り自動推定。
    列数不整合などの壊れ行はスキップ。欠損は線形補間 → 標準化。
    """
    base = os.path.basename(path).lower()
    if base == "lorenz.csv":
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python",
                         comment="#", usecols=[0])
    else:
        try:
            df = pd.read_csv(path, header=None, sep=",", usecols=[0],
                             comment="#", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(path, header=None, sep=None, engine="python",
                                 comment="#", usecols=[0], on_bad_lines="skip")
            except Exception:
                # 最後の手段：手動で先頭トークンだけ拾う
                import re as _re
                vals = []
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tok = _re.split(r"[\s,;]+", line, maxsplit=1)[0]
                        vals.append(tok)
                s = pd.to_numeric(pd.Series(vals), errors="coerce")
                s = s.interpolate(limit_direction="both").bfill().ffill()
                x = s.to_numpy(dtype=float)
                mu, sd = np.mean(x), np.std(x)
                return (x - mu) / sd if sd > 0 else (x - mu)

    if df.shape[1] < 1:
        raise ValueError("入力に列がありません（1列以上必要）")

    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s = s.interpolate(limit_direction="both").bfill().ffill()
    x = s.to_numpy(dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / sd if sd > 0 else (x - mu)

def recurrence_matrix(X, epsilon, normalize=True):
    X = np.asarray(X, dtype=float)
    if normalize:
        s = np.std(X)
        Xn = X / s if s > 0 else X.copy()
    else:
        Xn = X
    D = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=2))
    R = (D <= epsilon).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

def diag_line_lengths(R):
    N = R.shape[0]
    lengths = []
    for k in range(-(N-1), N):
        diag = np.diag(R, k=k)
        if diag.size == 0:
            continue
        run = 0
        for v in diag:
            if v == 1:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.array(lengths, dtype=int) if lengths else np.array([], dtype=int)

def rqa_measures(R, l_min=2):
    lengths = diag_line_lengths(R)
    RR_points = int(R.sum())
    if lengths.size == 0 or RR_points == 0:
        return 0.0, 0.0, 0.0
    sel = lengths[lengths >= l_min]
    if sel.size == 0:
        return 0.0, 0.0, 0.0
    det = float(np.sum(sel) / RR_points)
    L = float(np.mean(sel))
    _, counts = np.unique(sel, return_counts=True)
    p = counts / counts.sum()
    EN = float(-np.sum(p * np.log(p + 1e-12)))
    return det, L, EN

def vertical_line_lengths(R: np.ndarray) -> np.ndarray:
    """1 の連続縦列長をすべて返す"""
    n = R.shape[0]
    lengths = []
    for j in range(n):
        col = R[:, j]
        run = 0
        for v in col:
            if v == 1:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.array(lengths, dtype=int) if lengths else np.array([], dtype=int)

def white_vertical_line_lengths(R: np.ndarray) -> np.ndarray:
    """0 の連続縦列長（“白縦線”）を返す。対角の 0 は既に入っている前提。"""
    n = R.shape[0]
    lengths = []
    for j in range(n):
        col = R[:, j]
        run = 0
        for v in col:
            if v == 0:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.array(lengths, dtype=int) if lengths else np.array([], dtype=int)

def entropy_from_lengths(lengths: np.ndarray) -> float:
    if lengths.size == 0:
        return 0.0
    _, counts = np.unique(lengths, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))

def rqa_measures_full(R: np.ndarray, l_min=2, v_min=2, w_min=2) -> dict:
    """
    RP から “主要全指標” を一括計算して dict で返す。
    - RR: recurrence rate
    - DET: determinism
    - L, Lmax, L_entr, DIV
    - LAM, TT, Vmax, V_entr
    - W, Wmax, W_div, w_entr
    """
    N = R.shape[0]
    if N == 0:
        return {}

    # RR
    rr = float(R.mean())

    # 対角線系
    dls = diag_line_lengths(R)
    dls_sel = dls[dls >= l_min]
    det = float(np.sum(dls_sel) / (R.sum() + 1e-12)) if dls_sel.size > 0 and R.sum() > 0 else 0.0
    L = float(np.mean(dls_sel)) if dls_sel.size > 0 else 0.0
    Lmax = int(dls_sel.max()) if dls_sel.size > 0 else 0
    L_entr = entropy_from_lengths(dls_sel)
    DIV = float(1.0 / Lmax) if Lmax > 0 else 0.0

    # 垂直線系
    vls = vertical_line_lengths(R)
    vls_sel = vls[vls >= v_min]
    LAM = float(np.sum(vls_sel) / (R.sum() + 1e-12)) if vls_sel.size > 0 and R.sum() > 0 else 0.0
    TT = float(np.mean(vls_sel)) if vls_sel.size > 0 else 0.0
    Vmax = int(vls_sel.max()) if vls_sel.size > 0 else 0
    V_entr = entropy_from_lengths(vls_sel)

    # 白縦線系（R==0 の連続長）
    wls = white_vertical_line_lengths(R)
    wls_sel = wls[wls >= w_min]
    W = float(np.mean(wls_sel)) if wls_sel.size > 0 else 0.0
    Wmax = int(wls_sel.max()) if wls_sel.size > 0 else 0
    W_div = float(1.0 / Wmax) if Wmax > 0 else 0.0
    w_entr = entropy_from_lengths(wls_sel)

    return {
        "RR": rr,
        "DET": det,
        "L": L, "L_max": Lmax, "DIV": DIV, "L_entr": L_entr,
        "LAM": LAM, "TT": TT, "V_max": Vmax, "V_entr": V_entr,
        "W": W, "W_max": Wmax, "W_div": W_div, "w_entr": w_entr,
    }

def pretty_print_full_measures(meas: dict, lmin: int, vmin: int, wmin: int):
    print("\nRQA Result (full, self-computed)")
    print("=================================")
    print(f"L_min={lmin}, V_min={vmin}, W_min={wmin}")
    for k in ["RR","DET","L","L_max","DIV","L_entr","LAM","TT","V_max","V_entr","W","W_max","W_div","w_entr"]:
        v = meas.get(k, float("nan"))
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {float(v):.6f}")


def epsilon_for_target_rr(X, target_rr=0.05, normalize=True):
    X = np.asarray(X, float)
    if normalize and np.std(X) > 0:
        X = X / np.std(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    flat = D[np.isfinite(D)]
    k = max(1, int(target_rr * flat.size))
    return float(np.partition(flat, k)[k])

def compute_rqa_from_tau_m(series, tau, m, epsilons, l_min=2):
    """
    τ・m から RQA を計算。m は安全上限でクリップしてから _embed。
    """
    if tau < 1:
        tau = 1
    max_m_allowed = 1 + (len(series) - 2) // max(1, tau)
    m_use = int(max(2, min(m, min(MAX_DIM, max_m_allowed))))  # ★ 上限は MAX_DIM=10
    X = _embed(series, m_use, tau)
    if X.size == 0:
        return [], [], [], {"m_use": m_use, "mD": np.nan, "mL": np.nan, "mEN": np.nan}, X

    det_list, L_list, EN_list = [], [], []
    for eps in epsilons:
        R = recurrence_matrix(X, epsilon=eps, normalize=True)
        det, L, EN = rqa_measures(R, l_min=l_min)
        det_list.append(det); L_list.append(L); EN_list.append(EN)

    mD = float(np.mean(det_list)) if det_list else np.nan
    mL = float(np.mean(L_list))  if L_list  else np.nan
    mEN = float(np.mean(EN_list)) if EN_list else np.nan
    return det_list, L_list, EN_list, {"m_use": m_use, "mD": mD, "mL": mL, "mEN": mEN}, X

def pretty_print_pyrqa_result(res):
    def g(name, default=float("nan")):
        return getattr(res, name, default)

    print("\nRQA Result:")
    print("===========")
    print(f"Minimum diagonal line length (L_min): {int(g('minimal_diagonal_line_length', 2))}")
    print(f"Minimum vertical line length (V_min): {int(g('minimal_vertical_line_length', 2))}")
    print(f"Minimum white vertical line length (W_min): {int(g('minimal_white_vertical_line_length', 2))}\n")
    print(f"Recurrence rate (RR): {g('recurrence_rate'):.6f}")
    print(f"Determinism (DET): {g('determinism'):.6f}")
    print(f"Average diagonal line length (L): {g('average_diagonal_line'):.6f}")
    print(f"Longest diagonal line length (L_max): {int(g('longest_diagonal_line', 0))}")
    print(f"Divergence (DIV): {g('divergence'):.6f}")
    print(f"Entropy diagonal lines (L_entr): {g('entropy_diagonal_lines'):.6f}")
    print(f"Laminarity (LAM): {g('laminarity'):.6f}")
    print(f"Trapping time (TT): {g('trapping_time'):.6f}")
    print(f"Longest vertical line length (V_max): {int(g('longest_vertical_line', 0))}")
    print(f"Entropy vertical lines (V_entr): {g('entropy_vertical_lines'):.6f}")
    print(f"Average white vertical line length (W): {g('average_white_vertical_line', float('nan')):.6f}")
    print(f"Longest white vertical line length (W_max): {int(g('longest_white_vertical_line', 0))}")
    print(f"Longest white vertical line inverse (W_div): {g('longest_white_vertical_line_inverse', float('nan')):.6f}")
    print(f"Entropy white vertical lines (w_entr): {g('entropy_white_vertical_lines', float('nan')):.6f}")
    rr, det, lam = g('recurrence_rate'), g('determinism'), g('laminarity')
    if rr and rr > 0: print(f"\nRatio determinism / recurrence rate (DET/RR): {det/rr:.6f}")
    if det and det > 0: print(f"Ratio laminarity / determinism (LAM/DET): {lam/det:.6f}")


# ========= メイン =========
def main():
    ap = argparse.ArgumentParser(description="RQA from estimated tau & minimal m (filename only)")
    ap.add_argument("data_path", help="解析するデータファイル（1列目のみ解析）")
    ap.add_argument("--eps_min", type=float, default=0.1)
    ap.add_argument("--eps_max", type=float, default=2.0)
    ap.add_argument("--eps_steps", type=int, default=20)
    ap.add_argument("--lmin", type=int, default=2)
    ap.add_argument("--save_rp", action="store_true", help="リカレンスプロット（RR≈5%）も保存する")
    args = ap.parse_args()

    # 1) 読み込み（1列目のみ・補間・標準化）
    series = load_series_first_col_auto(args.data_path)

    # 2) τ の決め方：lorenz.csv は固定 5、その他は推定（max_lag=100）
    is_lorenz = os.path.basename(args.data_path).lower() == "lorenz.csv"
    tau = 5 if is_lorenz else determine_tau(series, max_lag=MAX_LAG_TAU)

    # 3) E1→最小 m の推定（max_dim=10, k=50%）
    k = int(max(1, K_FRAC * len(series)))
    E1 = itho_e1(series, max_dim=MAX_DIM, tau=tau, s=tau, k=k, theiler=int(tau))
    
    m_est = estimate_min_dimension(E1, eps=0.05, win=3)
    m_est = int(min(m_est, MAX_DIM))  # ★ 上限10にクリップ

    # 4) RQA 実行（εレンジ）
    epsilons = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    det_list, L_list, EN_list, avg, X = compute_rqa_from_tau_m(series, tau, m_est, epsilons, l_min=args.lmin)

    # 5) 出力保存
    base = sanitize_filename(args.data_path)
    out_dir = f"RQA_2{base}"
    os.makedirs(out_dir, exist_ok=True)

    # --- 5a) RQA 曲線の CSV 出力（εごと） ---
    if det_list:
        curves_df = pd.DataFrame({
            "epsilon": epsilons,
            "Determinism": det_list,
            "LineLength": L_list,
            "Entropy": EN_list,
        })
        curves_csv_path = os.path.join(out_dir, f"rqa_curves_{base}.csv")
        curves_df.to_csv(curves_csv_path, index=False, encoding="utf-8")
        print(f"[Saved] {curves_csv_path}")

    # --- 5b) グラフ分離: Determinism は単独、他2指標は同一図 ---
    if det_list:
        # (i) Determinism 単独
        plt.figure(figsize=(8, 3.6))
        plt.plot(epsilons, det_list, label='Determinism')
        title_tau = f"tau={'5 (fixed)' if is_lorenz else str(tau)}"
        plt.title(f"Determinism vs ε ({title_tau}, m={avg['m_use']})")
        plt.xlabel("Threshold ε"); plt.ylabel("Determinism")
        plt.legend(); plt.tight_layout()
        det_curve_path = os.path.join(out_dir, f"determinism_curve_{base}.png")
        plt.savefig(det_curve_path, dpi=200); plt.close()
        print(f"[Saved] {det_curve_path}")

        # (ii) 残り2指標（Line Length, Entropy）のみ
        plt.figure(figsize=(10, 4))
        plt.plot(epsilons, L_list, label='Line Length')
        plt.plot(epsilons, EN_list, label='Entropy')
        plt.title(f"RQA vs ε (no DET) ({title_tau}, m={avg['m_use']})")
        plt.xlabel("Threshold ε"); plt.ylabel("RQA Measure")
        plt.legend(); plt.tight_layout()
        curve_path = os.path.join(out_dir, f"rqa_curves_no_det_{base}.png")
        plt.savefig(curve_path, dpi=200); plt.close()
        print(f"[Saved] {curve_path}")

    # RP（任意）
    if args.save_rp and X.size > 0:
        try:
            eps_rp = epsilon_for_target_rr(X, target_rr=0.05, normalize=True)
            R_rp = recurrence_matrix(X, epsilon=eps_rp, normalize=True)
            plt.figure(figsize=(5,5))
            plt.imshow(R_rp, origin='lower', cmap='gray_r', interpolation='none')
            plt.title(f"RP (ε≈{eps_rp:.3g}, RR={R_rp.mean():.3f})")
            plt.xlabel("i"); plt.ylabel("j"); plt.tight_layout()
            rp_path = os.path.join(out_dir, f"rp_{base}.png")
            plt.savefig(rp_path, dpi=200); plt.close()
            print(f"[Saved] {rp_path}")
        except Exception as e:
            print(f"[WARN] RP save failed: {e}")
        # --- 5d) RR≈5% の ε で RP を作って、全指標を自前計算して出力＆CSV保存 ---
    if X.size > 0:
        try:
            eps_full = epsilon_for_target_rr(X, target_rr=0.05, normalize=True)
            R5 = recurrence_matrix(X, epsilon=eps_full, normalize=True)
            full = rqa_measures_full(R5, l_min=args.lmin, v_min=args.lmin, w_min=args.lmin)
            pretty_print_full_measures(full, args.lmin, args.lmin, args.lmin)

            full_df = pd.DataFrame([{
                "file": os.path.basename(args.data_path),
                "tau": 5 if is_lorenz else int(tau),
                "m_use": int(avg["m_use"]) if np.isfinite(avg["m_use"]) else None,
                "epsilon_rr5": eps_full,
                **full
            }])
            full_csv_path = os.path.join(out_dir, f"full_measures_rr5_{base}.csv")
            full_df.to_csv(full_csv_path, index=False, encoding="utf-8")
            print(f"[Saved] {full_csv_path}")
        except Exception as e:
            print(f"[WARN] Full-measure calc failed: {e}")

    
    # --- 5c) （任意）PyRQA があれば、εごとに“主要全指標”をCSV保存 ---

    if _HAS_PYRQA and X.size > 0:
        # pyrqa は内部で埋め込みも行えるが、ここではシリーズと (m_use, tau) を指定
        # 注意: εは FixedRadius の半径として使う
        py_rows = []
        for eps in epsilons:
            try:
                ts = TimeSeries(series.tolist(), embedding=Embedding(dimension=int(avg["m_use"]), delay=int(tau)))
                settings = Settings(
                    time_series=ts,
                    analysis_type=Classic,
                    neighbourhood=FixedRadius(eps),
                    similarity_measure=EuclideanMetric
                )
                comp = RQAComputation.create(settings)
                res = comp.run()  # 各種 measure を持つ

                # 取得できる主な属性（バージョンにより差異あり）
                row = {
                    "epsilon": float(eps),
                    "recurrence_rate": getattr(res, "recurrence_rate", float("nan")),
                    "determinism": getattr(res, "determinism", float("nan")),
                    "average_diagonal_line": getattr(res, "average_diagonal_line", float("nan")),
                    "longest_diagonal_line": getattr(res, "longest_diagonal_line", float("nan")),
                    "entropy_diagonal_lines": getattr(res, "entropy_diagonal_lines", float("nan")),
                    "laminarity": getattr(res, "laminarity", float("nan")),
                    "trapping_time": getattr(res, "trapping_time", float("nan")),
                    "longest_vertical_line": getattr(res, "longest_vertical_line", float("nan")),
                    "average_vertical_line": getattr(res, "average_vertical_line", float("nan")),
                    "divergence": getattr(res, "divergence", float("nan")),
                }
                py_rows.append(row)
            except Exception as e:
                py_rows.append({"epsilon": float(eps), "error": str(e)})

        py_df = pd.DataFrame(py_rows)
        py_csv_path = os.path.join(out_dir, f"pyRQA_measures_{base}.csv")
        py_df.to_csv(py_csv_path, index=False, encoding="utf-8")
        print(f"[Saved] {py_csv_path}")
    elif not _HAS_PYRQA:
        print("[INFO] PyRQA (pyrqa) が見つからないため、pyRQA の全指標CSVはスキップしました。")

            # --- PyRQA で RR≈5% の ε を選び、スクショ風の一覧を表示 ---
    if _HAS_PYRQA and X.size > 0:
        try:
            eps_for_text = epsilon_for_target_rr(X, target_rr=0.05, normalize=True)
            ts = TimeSeries(series.tolist(), embedding=Embedding(dimension=int(avg["m_use"]), delay=int(tau)))
            settings = Settings(
                time_series=ts,
                analysis_type=Classic,
                neighbourhood=FixedRadius(eps_for_text),
                similarity_measure=EuclideanMetric
            )
            res = RQAComputation.create(settings).run()
            print(f"\n[PyRQA] epsilon for ~5% RR: {eps_for_text:.6f}")
            pretty_print_pyrqa_result(res)
        except Exception as e:
            print(f"[WARN] Pretty print failed: {e}")



    # 6) 平均値を表示＆CSV保存
    print("\n==== RQA RESULT OF AVERAGE ====")
    print(f"File: {os.path.basename(args.data_path)}")
    print(f"tau (delay time): {'5 (fixed)' if is_lorenz else tau}")
    print(f"m (embedded dimension): {avg['m_use']}")
    print(f"Determinism (mD): {avg['mD']:.3f}")
    print(f"Line Length (mL): {avg['mL']:.3f}")
    print(f"Entropy (mEN): {avg['mEN']:.3f}")



    df = pd.DataFrame([{
        "file": os.path.basename(args.data_path),
        "tau": 5 if is_lorenz else int(tau),
        "m_use": int(avg["m_use"]) if np.isfinite(avg["m_use"]) else None,
        "mD": avg["mD"], "mL": avg["mL"], "mEN": avg["mEN"]
    }])
    csv_path = os.path.join(out_dir, f"result_summary_{base}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[Saved] {csv_path}")

if __name__ == "__main__":
    main()
