"""既知系 → ブロック欠損 → 線形補間 → IAAFT → 佐野沢田λ。
元コードの力学系・欠損規則・τ/m推定・佐野沢田法を維持。
Z-score/有意性検定は行わず、λと生成品質を記録する。
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score

# --- 力学系を生成 ---
def generate_brownian(n_steps=20000, D=1.0, dt=1.0):
    """
    1次元ブラウン運動（ランダムウォーク）
    x[0] = 0 から始めて、ガウス乱数を積分していく。
    D: 拡散係数
    dt: 時間刻み（ここでは便宜的に 1.0）
    """
    x = np.zeros(n_steps)
    # 増分 dW ~ N(0, 2 D dt)
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


# ====== 破損線形補間を施している ======
def introduce_block_missing_and_interpolate(x, missing_rate=0.1, block_len=10, seed=None, pattern = "random"):
    """
    連続ブロック欠損をランダムに付与し、線形補間で埋める。
    missing_rate : 全体の何割を欠損させるか (0 <= r < 1)
    block_len    : 1ブロックの欠損長（サンプル数）
    """
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
        
        #上位40%を候補とする（元コードの60パーセンタイルを維持）
        thresh = np.percentile(diff, 60)

        candidates = np.where(diff >= thresh)[0]
    
    elif pattern == "low":

        #下位40%を候補とする（元コードの40パーセンタイルを維持）
        thresh = np.percentile(diff, 40)

        candidates = np.where(diff <= thresh)[0]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    candidates = candidates[candidates < n - block_len]

    if len(candidates) == 0:
        raise ValueError("No candidate positions for missing blocks")

    #ブロック配置
    placed = 0
    trial = 0
    max_trial = 20000

    while placed < num_blocks and trial < max_trial:
        
        start = rng.choice(candidates)

        if not mask[start:start + block_len].any():
            mask[start:start + block_len] = True
            placed += 1
        
        trial += 1


    # 上限調整
    if mask.sum() > total_missing:
        extra = mask.sum() - total_missing
        idx_true = np.where(mask)[0]
        remove_idx = rng.choice(idx_true, size=extra, replace=False)
        mask[remove_idx] = False


    y_missing = x.copy()
    y_missing[mask] = np.nan

    # 線形補間（端点は最近傍値で補完）
    xi = np.arange(n)
    valid = ~np.isnan(y_missing)
    y_filled = np.interp(
        xi,
        xi[valid],
        y_missing[valid]
    )
    print(
    f"[{pattern}] "
    f"candidates={len(candidates)} "
    f"placed={placed}/{num_blocks} "
    f"actual_missing={mask.sum()/n:.3f}"
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


# --- Sano-Sawada法：QR分解によるLyapunov spectrum推定 ---
def sano_sawada_lyapunov(
    data,
    m=3,
    tau=1,
    n_neighbors=30,
    theiler=None,
    step=1,
    dt=1.0,
    return_spectrum=False
):
    """
    Sano-Sawada法に基づき、
    局所線形写像を近傍点から最小二乗推定し、
    QR分解による接ベクトルの逐次伝播から
    Lyapunov spectrumを推定する。

    Parameters
    ----------
    data : array-like
        1次元時系列

    m : int
        埋め込み次元

    tau : int
        遅れ時間

    n_neighbors : int
        局所線形写像の推定に使用する近傍点数

    theiler : int or None
        Theiler window
        Noneなら tau*m

    step : int
        局所写像の時間ステップ

    dt : float
        元時系列のサンプリング時間
        Logistic mapなら1.0でよい

    return_spectrum : bool
        Trueなら全Lyapunov spectrumを返す
        Falseなら最大Lyapunov指数のみ返す
    """

    data = np.asarray(data, dtype=float)

    # ============================================
    # 1. 遅延座標によるアトラクタ再構成
    # X_i = [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]
    # ============================================
    embedded = _embed(data, m, tau)

    num_points = len(embedded)

    if num_points <= m + step:
        if return_spectrum:
            return np.full(m, np.nan)
        return np.nan

    if theiler is None:
        theiler = tau * m

    # ============================================
    # 2. 接空間の初期直交基底
    # ============================================
    Q = np.eye(m)

    # 各方向のlog伸長率を累積
    log_growth = np.zeros(m)

    valid_steps = 0

    # ============================================
    # 3. 軌道に沿って局所Jacobianを推定
    # ============================================
    for i in range(num_points - step):

        # 現在点との差
        diff = embedded[:num_points-step] - embedded[i]

        dist = np.linalg.norm(diff, axis=1)

        # 自分自身を除外
        dist[i] = np.inf

        # ========================================
        # Theiler window
        # 時間的に近すぎる点を除外
        # ========================================
        start = max(0, i - theiler)
        end = min(num_points-step, i + theiler + 1)

        dist[start:end] = np.inf

        # 有効な候補
        valid_idx = np.where(np.isfinite(dist))[0]

        if len(valid_idx) < m:
            continue

        # 近傍点
        nearest_idx = valid_idx[
            np.argsort(dist[valid_idx])[:n_neighbors]
        ]

        # step後まで存在する点のみ使用
        nearest_idx = nearest_idx[
            nearest_idx + step < num_points
        ]

        if len(nearest_idx) < m:
            continue

        # ========================================
        # 4. 近傍変位ベクトル
        #
        # y = X_k - X_i
        # z = X_{k+step} - X_{i+step}
        # ========================================
        Y = (
            embedded[nearest_idx]
            - embedded[i]
        )

        Z = (
            embedded[nearest_idx + step]
            - embedded[i + step]
        )

        try:

            # ====================================
            # 5. 局所線形写像 A を最小二乗推定
            #
            # 行ベクトル表現では
            #
            # Y @ B ≈ Z
            #
            # column-vector形式なら
            #
            # z = A y
            #
            # なので A = B.T
            # ====================================
            B, residuals, rank, s = np.linalg.lstsq(
                Y,
                Z,
                rcond=None
            )

            # rank不足なら局所写像を信用しない
            if rank < m:
                continue

            A = B.T

            # NaN / infチェック
            if not np.all(np.isfinite(A)):
                continue

            # ====================================
            # 6. 接ベクトルを時間発展
            #
            # Q_{j+1}' = A_j Q_j
            # ====================================
            propagated = A @ Q

            # ====================================
            # 7. QR分解
            #
            # propagated = Q_new R
            #
            # Gram-Schmidt直交化と同等の役割
            # ====================================
            Q_new, R = np.linalg.qr(propagated)

            diag_R = np.abs(np.diag(R))

            # log(0)防止
            if np.any(diag_R <= 1e-14):
                continue

            # ====================================
            # 8. 各方向の伸長率を累積
            # ====================================
            log_growth += np.log(diag_R)

            Q = Q_new

            valid_steps += 1

        except np.linalg.LinAlgError:
            continue

    # ============================================
    # 9. Lyapunov指数
    #
    # λ_i =
    # Σ log |R_ii| / total time
    # ============================================
    if valid_steps == 0:

        if return_spectrum:
            return np.full(m, np.nan)

        return np.nan

    total_time = valid_steps * step * dt

    lyapunov_spectrum = log_growth / total_time

    # 大きい順に並べる
    lyapunov_spectrum = np.sort(
        lyapunov_spectrum
    )[::-1]

    if return_spectrum:
        return lyapunov_spectrum

    # 従来コードとの互換性のため
    # 最大Lyapunov指数だけ返す
    return lyapunov_spectrum[0]


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


def acf(x, max_lag=100):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.full(min(max_lag, len(x)-1)+1, np.nan)
    # 非循環ACF：端点の影響も含むため、FFT一致だけでは完全一致しない
    return np.array([np.dot(x[:len(x)-k], x[k:]) / denom
                     for k in range(min(max_lag, len(x)-1)+1)])


def validate_iaaft(original, surrogate):
    x, y = np.asarray(original), np.asarray(surrogate)
    if x.shape != y.shape:
        raise ValueError("Shape mismatch")
    ax, ay = acf(x), acf(y)
    return dict(distribution_error=float(np.max(np.abs(np.sort(x)-np.sort(y)))),
                spectrum_error=spectrum_error(x, y),
                acf_rmse=float(np.sqrt(np.mean((ax-ay)**2))),
                same_as_input=bool(np.array_equal(x, y)))


def plot_validation(x, s, info, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x[:500], label="Analysis input", alpha=.8)
    axes[0, 0].plot(s[:500], label="IAAFT", alpha=.6)
    axes[0, 0].set_title("First 500 samples")
    bins = np.histogram_bin_edges(x, bins=30)
    axes[0, 1].hist(x, bins=bins, histtype="step", label="Input")
    axes[0, 1].hist(s, bins=bins, histtype="step", linestyle="--", label="IAAFT")
    axes[0, 1].set_title("Value distribution")
    freq = np.fft.rfftfreq(len(x))
    for v, label in [(x, "Input"), (s, "IAAFT")]:
        power = np.abs(np.fft.rfft(v))**2 / len(v)
        axes[1, 0].semilogy(freq[1:], np.maximum(power[1:], 1e-30), label=label)
        axes[1, 1].plot(acf(v), label=label)
    axes[1, 0].set_title("Periodogram (DC excluded)")
    axes[1, 0].set_xlabel("Frequency [cycles/sample]")
    axes[1, 1].set_title("Non-circular ACF")
    for ax in axes.flat:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(f"IAAFT: spectral error={info['spectrum_error']:.3e}; {info['stop_reason']}")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def run(args):
    if args.n_steps < 200 or args.num_surrogates < 1 or args.max_iter < 1 or args.tol < 0:
        raise ValueError("n_steps >= 200, num_surrogates/max_iter >= 1, tol >= 0 required")
    if any(not 0 <= r < 1 for r in args.missing_rates):
        raise ValueError("Missing rates must be in [0,1)")
    # 元の確率系生成器も再現可能にする。IAAFT用乱数とは分離。
    np.random.seed(args.seed)
    generators = {"brown": generate_brownian, "lorenz": generate_lorenz,
                  "logistic": generate_logistic, "sine": generate_sin,
                  "white": generate_white_noise}
    out = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    summary = []
    for system in args.systems:
        base = generators[system](n_steps=args.n_steps)
        for rate in args.missing_rates:
            patterns = ["random"] if rate == 0 else ["random", "high", "low"]
            for pattern_id, pattern in enumerate(patterns):
                folder = out / system / f"rate_{rate:g}" / f"pattern_{pattern}"
                folder.mkdir(parents=True)
                missing, data, mask = introduce_block_missing_and_interpolate(
                    base, rate, block_len=10, seed=pattern_id, pattern=pattern)
                pd.DataFrame(dict(time=np.arange(len(base)), original=base,
                                  missing=missing, interpolated=data, mask=mask.astype(int))).to_csv(
                                      folder / "timeseries.csv", index=False)
                tau = int(np.clip(determine_tau(data, max_lag=min(100, len(data)//4)), 1, 50))
                e1 = itho_e1(data, max_dim=10, tau=tau, theiler=tau*2)
                m, m_status = 2, "fallback_2_no_plateau"
                for i in range(1, len(e1)):
                    if np.isfinite(e1[i-1]) and e1[i-1] > 0 and abs(e1[i]/e1[i-1]-1) < .05:
                        m, m_status = i+1, "plateau_found"
                        break
                pd.DataFrame({"dimension": np.arange(1, len(e1)+1), "E1": e1}).to_csv(folder / "E1.csv", index=False)
                real = sano_sawada_lyapunov(data, m=m, tau=tau, theiler=tau*m, dt=1.0)
                print(f"{system}, rate={rate}, pattern={pattern}, tau={tau}, m={m}: input lambda={real}", flush=True)
                rows = []
                for j in range(args.num_surrogates):
                    seed = int(np.random.SeedSequence([args.seed, list(generators).index(system),
                                                      args.missing_rates.index(rate), pattern_id, j]).generate_state(1)[0])
                    surrogate, info = iaaft_surrogate(data, args.max_iter, args.tol, seed, True)
                    metrics = validate_iaaft(data, surrogate)
                    lam = sano_sawada_lyapunov(surrogate, m=m, tau=tau, theiler=tau*m, dt=1.0)
                    rows.append(dict(surrogate_id=j+1, seed=seed, lambda_iaaft=lam,
                                     iterations=info["iterations"], stop_reason=info["stop_reason"], **metrics))
                    if args.save_surrogates or j == 0:
                        pd.DataFrame({"input": data, "iaaft": surrogate}).to_csv(folder / f"surrogate_{j+1:03d}.csv", index=False)
                    if j == 0:
                        plot_validation(data, surrogate, info, folder / "IAAFT_validation.png")
                        pd.DataFrame({"iteration": np.arange(1,len(info['error_history'])+1),
                                      "spectrum_error": info['error_history']}).to_csv(folder / "convergence_first.csv", index=False)
                    pd.DataFrame(rows).to_csv(folder / "IAAFT_results.csv", index=False)
                    print(f"  IAAFT {j+1}/{args.num_surrogates}: lambda={lam:.6g}, error={metrics['spectrum_error']:.3e}, {info['stop_reason']}", flush=True)
                frame = pd.DataFrame(rows)
                finite = frame.loc[np.isfinite(frame.lambda_iaaft), "lambda_iaaft"]
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(finite, bins=15, color="skyblue", edgecolor="black", label="IAAFT")
                if np.isfinite(real):
                    ax.axvline(real, color="red", linestyle="--", label=f"Input lambda={real:.4g}")
                ax.set(title=f"{system}, requested={rate:.0%}, actual={mask.mean():.1%}, {pattern}",
                       xlabel=r"Maximum Lyapunov exponent $\lambda$ [per sample]", ylabel="Count")
                ax.legend(); fig.tight_layout(); fig.savefig(folder / "IAAFT_lambda_histogram.png", dpi=150); plt.close(fig)
                if rate > 0:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(base, linewidth=.6, label="Original")
                    ax.scatter(np.flatnonzero(mask), data[mask], s=2, color="red", label="Interpolated")
                    ax.legend(); fig.tight_layout(); fig.savefig(folder / "missing_visualization.png", dpi=150); plt.close(fig)
                summary.append(dict(system=system, requested_missing_rate=rate, actual_missing_rate=float(mask.mean()),
                                    pattern=pattern, tau=tau, m=m, m_status=m_status, lambda_input=real,
                                    num_surrogates=args.num_surrogates, valid_lambda_count=len(finite),
                                    lambda_surrogate_mean=finite.mean(), lambda_surrogate_std=finite.std(ddof=1),
                                    max_distribution_error=frame.distribution_error.max(),
                                    mean_spectrum_error=frame.spectrum_error.mean(), max_spectrum_error=frame.spectrum_error.max(),
                                    mean_acf_rmse=frame.acf_rmse.mean(), max_iter_count=int((frame.stop_reason=='max_iter').sum()),
                                    same_as_input_count=int(frame.same_as_input.sum())))
                pd.DataFrame(summary).to_csv(out / "IAAFT_summary.csv", index=False)
    print(f"Saved: {out.resolve()}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=20000)
    p.add_argument("--num-surrogates", type=int, default=39)
    p.add_argument("--systems", nargs="+", choices=["brown", "lorenz", "logistic", "sine", "white"],
                   default=["brown", "lorenz", "logistic", "sine", "white"])
    p.add_argument("--missing-rates", nargs="+", type=float, default=[0,.1,.3,.5,.7])
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--output-dir", default="result_IAAFT_linear_defect")
    p.add_argument("--save-surrogates", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
