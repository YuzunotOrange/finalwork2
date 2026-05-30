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
