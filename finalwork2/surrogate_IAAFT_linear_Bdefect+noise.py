import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score

def generate_brownian(n_steps=20000, D=1.0, dt=1.0):
    x = np.zeros(n_steps)

    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    dW = sqrt_2Ddt * np.random.randn(n_steps - 1)
    x[1:] = np.cumsum(dW)
    return x

def generate_lorenz(n_steps=20000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    for i in range(n_steps-1):
        dx = sigma*(y[i]-x[i])
        dy = x[i]*(rho - z[i]) - y[i]
        dz = x[i]*y[i] - beta*z[i]
        x[i+1] = x[i] + dx*dt
        y[i+1] = y[i] + dy*dt
        z[i+1] = z[i] + dz*dt
    return x

def generate_sin(n_steps=20000, freq=1.0, dt=0.01):
    t = np.arange(n_steps) * dt
    return np.sin(2*np.pi*freq*t)

def generate_logistic(n_steps=20000, r=4.0, x0=0.3, discard=1000):
    x = np.zeros(n_steps+discard)
    x[0] = x0
    for i in range(n_steps+discard-1):
        x[i+1] = r*x[i]*(1-x[i])
    return x[discard:]

def generate_white_noise(n_steps=20000):
    return np.random.normal(0, 1, n_steps)

def add_poisson_observation_noise(x, strength=0.1, seed=None):
    """
    strength:
        ノイズの期待RMS / 元時系列の標準偏差。
        0ならノイズなし、0.1なら約10%、0.5なら約50%。

    時系列を条件として E[y_i] = x_i を保つ観測ノイズ。
    各点のノイズ分散は信号値に依存する。
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a non-empty, finite 1-D series")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and >= 0")
    if strength == 0:
        return x.copy()

    scale = np.std(x)
    if scale == 0:
        raise ValueError(
            "Constant series has no standard deviation; "
            "use strength=0 or define an absolute noise scale."
        )

    # 最小値でも正の期待カウントを持つよう、背景レベル1を加える
    offset = np.min(x)
    positive_signal = (x - offset) / scale + 1.0

    # 平均ノイズ分散を (strength * scale)**2 に設定
    gain = np.mean(positive_signal) / strength**2
    expected_counts = gain * positive_signal

    # 極端に小さいstrengthによるオーバーフローを検出
    if (
        not np.all(np.isfinite(expected_counts))
        or np.max(expected_counts) > 1e18
    ):
        raise ValueError("strength is too small for Poisson sampling")

    rng = np.random.default_rng(seed)
    counts = rng.poisson(expected_counts)

    return offset + scale * (counts / gain - 1.0)

def introduce_block_missing_and_interpolate(x, missing_rate=0.1, block_len=10, seed=None, pattern = "random"):

    x = np.asarray(x, dtype=float)
    n = len(x)
    if missing_rate <= 0:
        return x.copy(), x.copy(), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)

    total_missing = int(round(missing_rate * n))
    num_blocks = max(1, total_missing // block_len)

    mask = np.zeros(n, dtype=bool)

    diff = np.abs(np.diff(x))

    if pattern == "random":
        candidates = np.arange(n - block_len)
    
    elif pattern == "high":
        

        thresh = np.percentile(diff, 60)

        candidates = np.where(diff >= thresh)[0]
    
    elif pattern == "low":


        thresh = np.percentile(diff, 40)

        candidates = np.where(diff <= thresh)[0]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    candidates = candidates[candidates < n - block_len]

    if len(candidates) == 0:
        raise ValueError("No candidate positions for missing blocks")


    placed = 0
    trial = 0
    max_trial = 20000

    while placed < num_blocks and trial < max_trial:
        
        start = rng.choice(candidates)

        if not mask[start:start + block_len].any():
            mask[start:start + block_len] = True
            placed += 1
        
        trial += 1



    if mask.sum() > total_missing:
        extra = mask.sum() - total_missing
        idx_true = np.where(mask)[0]
        remove_idx = rng.choice(idx_true, size=extra, replace=False)
        mask[remove_idx] = False


    y_missing = x.copy()
    y_missing[mask] = np.nan


    xi = np.arange(n)
    valid = ~np.isnan(y_missing)
    y_filled = np.interp(
        xi,
        xi[valid],
        y_missing[valid]
    )







    return y_missing, y_filled, mask


def average_mutual_information(x, max_lag=100, bins=32):

    x = np.asarray(x)

    ami = []

    for lag in range(1, max_lag + 1):

        x1 = x[:-lag]
        x2 = x[lag:]

        # ヒストグラム分割
        x1_bin = np.digitize(
            x1,
            np.histogram_bin_edges(x1, bins=bins)
        )

        x2_bin = np.digitize(
            x2,
            np.histogram_bin_edges(x2, bins=bins)
        )

        mi = mutual_info_score(x1_bin, x2_bin)

        ami.append(mi)

    return np.array(ami)

def determine_tau(series, max_lag=100):

    ami = average_mutual_information(
        series,
        max_lag=max_lag
    )

    # 最初の極小値
    for i in range(1, len(ami)-1):

        if ami[i] < ami[i-1] and ami[i] < ami[i+1]:
            return i + 1

    return np.argmin(ami) + 1

def _embed(x, m, tau):
    N = len(x) - (m - 1)*tau
    if N <= 0: return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau*np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    if s is None: s = tau
    if k is None: k = 10
    E1 = []
    for m in range(1, max_dim+1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0: break
        valid_M = M - s
        X_now = Xm[:valid_M]
        k_eff = min(k, valid_M-1)
        if k_eff < 1: break
        nn = NearestNeighbors(n_neighbors=min(k_eff+21, valid_M), algorithm='kd_tree').fit(X_now)
        dists, idxs = nn.kneighbors(X_now)
        ratios = []
        for i in range(valid_M):
            j = idxs[i, 1:]
            d = dists[i, 1:]
            if theiler > 0:
                mask = np.abs(j - i) > theiler
                j, d = j[mask], d[mask]
            if j.size == 0: continue
            mask2 = (j + s) < M
            j, d = j[mask2], d[mask2]
            if j.size == 0: continue
            use = min(k_eff, j.size)
            jj = j[:use]
            dn = np.mean(d[:use])
            if dn < 1e-12: continue
            df = np.mean(np.linalg.norm(Xm[i+s] - Xm[jj+s], axis=1))
            ratios.append(df/dn)
        if ratios: E1.append(np.mean(ratios))
    return np.array(E1)


# --- 旧佐野沢田実装：局所写像の最大特異値の対数を平均（QRなし） ---
def sano_sawada_lyapunov(data, m = 3, tau = 1, n_neighbors = 30, theiler=None):
    
    N = len(data)
    #アトラクタの再構築
    embedded = _embed(np.asarray(data, dtype=float), m, tau)
    num_points = len(embedded)

    lyap_exp = []
    #接ベクトルの時間発展を計算
    step = 1 #写像の場合は1ステップづつ、連続系なら間隔を調整
    
    for i in range(0, num_points - step, 10):
        search_range = num_points - step
        diff = embedded[:search_range] - embedded[i]
        
        dist = np.linalg.norm(diff, axis=1)

        # 自分自身を除外
        dist[i] = np.inf

        # Theiler Window
        if theiler is None:
            theiler = tau * m
        w = theiler
        start = max(0, i - w)
        end = min(search_range, i + w + 1)

        dist[start:end] = np.inf

        #近傍点のインデックスを取得
        valid_idx = np.flatnonzero(np.isfinite(dist))
        if len(valid_idx) < m:
            continue
        nearest_idx = valid_idx[np.argsort(dist[valid_idx])[:n_neighbors]]

        #ヤコビ行列の推定用行列構成
        z = embedded[nearest_idx] - embedded[i]
        z_next = embedded[nearest_idx + step] -embedded[i + step]

        try:
            #最小二乗法で A (z_next = A * z)を推定
            A, _, _, _ = np.linalg.lstsq(z, z_next, rcond=None)
            #特異値分解から最大伸長率を抽出
            singular_values = np.linalg.svd(A, compute_uv=False)
            if singular_values[0] > 0:
                lyap_exp.append(np.log(singular_values[0]))
        except np.linalg.LinAlgError:
            continue
    return np.mean(lyap_exp) if lyap_exp else np.nan


# --- IAAFTサロゲート生成 ---
def spectrum_error(x, y):
    """DCを除く両側Fourier振幅の相対L2誤差。"""
    a = np.abs(np.fft.fft(x))[1:]
    b = np.abs(np.fft.fft(y))[1:]
    norm = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / norm) if norm > 0 else 0.0


def iaaft_surrogate(x, max_iter=1000, tol=1e-8, seed=None, return_info=False):
    """分布補正を最後に行うIAAFT。tolは誤差改善量で、品質の合格閾値ではない。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 1-D series with at least 3 samples")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter >= 1 and tol >= 0 are required")
    rng = np.random.default_rng(seed)
    sorted_x = np.sort(x)
    target = np.fft.rfft(x)
    amplitude = np.abs(target)
    current = rng.permutation(x)
    best = current.copy()
    best_error = spectrum_error(x, best)
    previous_error = best_error
    history = []
    stagnant = 0
    reason = "max_iter"
    if np.ptp(x) == 0:
        info = dict(iterations=0, stop_reason="constant", spectrum_error=0.0,
                    error_history=[], same_as_input=True)
        return (current, info) if return_info else current
    for iteration in range(1, max_iter + 1):
        fft = np.fft.rfft(current)
        adjusted_fft = amplitude * np.exp(1j * np.angle(fft))
        adjusted_fft[0] = target[0]  # DCは元の符号も維持
        if len(x) % 2 == 0:
            adjusted_fft[-1] = amplitude[-1] * (1 if fft[-1].real >= 0 else -1)
        adjusted = np.fft.irfft(adjusted_fft, n=len(x))
        order = np.argsort(adjusted, kind="stable")
        updated = np.empty_like(x)
        updated[order] = sorted_x
        error = spectrum_error(x, updated)
        history.append(error)
        if error <= best_error:
            best, best_error = updated.copy(), error
        if np.array_equal(updated, current):
            reason = "fixed_point"
            break
        stagnant = stagnant + 1 if abs(previous_error - error) <= tol else 0
        current, previous_error = updated, error
        if stagnant >= 10:
            reason = "stagnation"
            break
    info = dict(iterations=iteration, stop_reason=reason,
                spectrum_error=best_error, error_history=history,
                same_as_input=bool(np.array_equal(x, best)))
    return (best, info) if return_info else best


def significance_label(z):
    if np.isnan(z):
        return "NaN"
    
    if abs(z) > 2.58:
        return "1%"
    
    elif abs(z) > 1.96:
        return "5%"
    
    else:
        return "NS"

def surrogate_z_test(original_value, surrogate_values):

    surrogate_values = np.asarray(surrogate_values)

    mu = np.mean(surrogate_values)
    sigma = np.std(surrogate_values, ddof=1)

    if sigma == 0:
        z = np.nan
    else:
        z = (original_value - mu) / sigma

    return {
        "z_score": z
    }





base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, "result_IAAFT_known_Bdefect(1987)+noise",
                          datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
os.makedirs(result_dir, exist_ok=True)
print(f"Result directory: {result_dir}")

num_runs = 5
num_surr = 39
n_steps = 20000
missing_rates = [0, .1, .3, .5, .7]
seed = 20261004

poisson_strength = 0.1  # 0: ノイズなし、0.1: 約10%、0.5: 約50%

generators = {
    "Brown_motion": generate_brownian,
    "Lorenz": generate_lorenz,
    "Logistic": generate_logistic,
    "Sine_wave": generate_sin,
    "White_noise": generate_white_noise,
}
datasets, originals, missing_series, masks = {}, {}, {}, {}
for system_id, (system, generator) in enumerate(generators.items()):
    np.random.seed(int(np.random.SeedSequence([seed, system_id]).generate_state(1)[0]))
    clean_base = generator(n_steps=n_steps)

    # 力学系ごとに独立した、再現可能なノイズ乱数を使用
    noise_seed = np.random.SeedSequence([seed, system_id, 314159])

    base = add_poisson_observation_noise(
        clean_base,
        strength=poisson_strength,
        seed=noise_seed,
    )

    for rate in missing_rates:
        patterns = ["random"] if rate == 0 else ["random", "high", "low"]
        for pattern_id, pattern in enumerate(patterns):
            missing, filled, mask = introduce_block_missing_and_interpolate(
                base, missing_rate=rate, block_len=10, seed=pattern_id, pattern=pattern)
            name = f"{system}_rate_{int(rate*100)}_{pattern}"
            datasets[name] = filled
            originals[name] = base
            missing_series[name] = missing
            masks[name] = mask

all_results = []


for name, base_data in datasets.items():

    file_safe_name = (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
    )

    # データごとの保存フォルダ
    system_dir = os.path.join(
        result_dir,
        file_safe_name
    )
    os.makedirs(system_dir, exist_ok=True)

    for run in range(1, num_runs + 1):

        print("\n==============================")
        print(f"Dataset : {name}")
        print(f"Run     : {run}/{num_runs}")
        print("==============================")

        # runごとの保存フォルダ
        run_dir = os.path.join(
            system_dir,
            f"run_{run}"
        )
        os.makedirs(run_dir, exist_ok=True)

        data = base_data.copy()

        # ===== 時系列データ保存 =====
        ts_df = pd.DataFrame({
            "time": np.arange(len(base_data)),
            "data": data,
            "original": originals[name],
            "missing": missing_series[name],
            "missing_mask": masks[name]
        })

        csv_path = os.path.join(
            run_dir,
            f"timeseries_{file_safe_name}.csv"
        )

        ts_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # ===== 遅れ時間 tau 推定 =====
        tau_est = determine_tau(data, max_lag=100)

        if tau_est < 1:
            tau_est = 1

        if tau_est > 50:
            tau_est = 50

        tau_val = tau_est

        print(f"Estimated Tau: {tau_val}")

        # ===== 埋め込み次元 m 推定 =====
        e1_values = itho_e1(
            data,
            max_dim=10,
            tau=tau_val,
            theiler=tau_val * 2
        )

        m_est = 2

        for i in range(1, len(e1_values)):

            ratio = e1_values[i] / e1_values[i - 1]

            if abs(ratio - 1.0) < 0.05:
                m_est = i + 1
                break

        print(f">> Estimated Parameters: Tau = {tau_val}, m = {m_est}")
        print(f"Estimated m: {m_est} (E1 values: {e1_values[:m_est]})")

        # ===== オリジナルデータのリアプノフ指数 =====
        print(
            f"Calculating original Lyapunov exponent "
            f"(m={m_est}, tau={tau_val})..."
        )

        real_lam = sano_sawada_lyapunov(
            data,
            m=m_est,
            tau=tau_val,
            theiler=tau_val * m_est
        )

        # ===== IAAFTサロゲート =====
        print("Calculating IAAFT Surrogates...")

        iaaft_lams = [
            sano_sawada_lyapunov(
                iaaft_surrogate(data),
                m=m_est,
                tau=tau_val,
                theiler=tau_val * m_est
            )
            for _ in range(num_surr)
        ]

        # ===== 結果CSV保存 =====
        result_df = pd.DataFrame({
            "type": (
                ["original"] +
                ["IAAFT_surrogate"] * num_surr
            ),
            "lambda": (
                [real_lam] +
                iaaft_lams
            )
        })

        result_csv_path = os.path.join(
            run_dir,
            f"lyapunov_results_{file_safe_name}.csv"
        )

        result_df.to_csv(result_csv_path, index=False)
        print(f"Saved result CSV: {result_csv_path}")

    # ===== Z-score calculation =====
        iaaft_result = surrogate_z_test(real_lam, iaaft_lams)

        all_results.append({

                "Data": name,
                "Run": run,
                "Original λ": round(real_lam, 6),

                "IAAFT Z-score": round(iaaft_result["z_score"], 3),
                "IAAFT Significance": significance_label(iaaft_result["z_score"]),

        })

        # ===== Summary CSVを逐次保存 =====
        summary_df = pd.DataFrame(all_results)

        summary_path = os.path.join(
            result_dir,
            "surrogate_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Updated Summary: {summary_path}")


        # ===== ヒストグラム作成 =====
        plt.figure(figsize=(10, 6))

        plt.hist(
            iaaft_lams,
            bins=15,
            color='skyblue',
            alpha=0.6,
            label='IAAFT Surrogates',
            edgecolor='blue'
        )

        plt.axvline(
            real_lam,
            color='red',
            linestyle='--',
            linewidth=3,
            label=rf'Original ($\lambda$={real_lam:.3f})'
        )

        plt.title(
            f"{name} "
            f"(Run {run}/{num_runs})"
        )

        plt.xlabel("Maximum Lyapunov Exponent ($\\lambda$)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        save_path = os.path.join(
            run_dir,
            f"result_{file_safe_name}.png"
        )

        plt.savefig(save_path)
        print(f"Saved figure: {save_path}")

        plt.close()

summary_df = pd.DataFrame(all_results)

summary_path = os.path.join(
     result_dir,
        "surrogate_summary.csv"
    )

summary_df.to_csv(
     summary_path,
       index=False
    )

print(f"Saved Summary: {summary_path}")

plt.show()
