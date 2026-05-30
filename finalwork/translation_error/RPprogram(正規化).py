import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import zipfile
import io # BytesIOを使用するために追加

# PyRQAモジュールのインポート
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator # このスクリプトでは直接使用されていません
from pyrqa.analysis_type import Classic

# --- 1. ヘルパー関数定義 ---

def calculate_autocorrelation(series, max_lag):
    """
    時系列データの自己相関関数を計算し、
    指定された最大遅れ時間までの自己相関値をリストで返します。
    """
    autocorr_values = []
    n = len(series)
    series_mean = np.mean(series)

    for lag in range(1, max_lag + 1):
        numerator = np.sum((series[:-lag] - series_mean) *
                           (series[lag:] - series_mean))
        denominator = np.sum((series - series_mean) ** 2)
        
        if denominator == 0:
            autocorr = 0.0
        else:
            autocorr = numerator / denominator
        autocorr_values.append(autocorr)
    return autocorr_values

# --- 2. データ読み込みと前処理 ---

file_path = r'C:\Users\bakus\OneDrive - SIT\研究解析-石川研究-\測定データ-鼻部\サカキバラ\サカキバラ_muon\sakakibara_muon.xlsx' 

file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
base_dir = os.path.dirname(file_path)

print(f"--- リカレンスプロット解析を開始します ---")
print(f"対象ファイル: {file_path}")
print(f"解析結果のZIPファイルおよび全RQA指標データ(Excel)は'{base_dir}'に保存されます。") # メッセージ変更

images_to_zip = []
all_rqa_metrics_data = [] 

try:
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("サポートされていないファイル形式です。Excel (.xlsx) または CSV (.csv) ファイルを指定してください。")

    if 'B-A' not in df.columns:
        raise ValueError("指定されたファイルに 'B-A' 列が見つかりません。")

    raw_data_full = df['B-A'].values

    if np.isnan(raw_data_full).any():
        nan_count = np.sum(np.isnan(raw_data_full))
        print(f"警告: データにNaN値が含まれています。{nan_count}個のNaN値を削除します。")
        raw_data_full = raw_data_full[~np.isnan(raw_data_full)]
        if len(raw_data_full) == 0:
            raise ValueError("NaN値を除去した結果、データが空になりました。")

    print(f"\n--- データ範囲設定 ---")
    print(f"元の全データ長: {len(raw_data_full)}点")
    
    while True:
        try:
            start_index = int(input(f"開始インデックス を入力してください (0から{len(raw_data_full)-1}の間): "))
            if 0 <= start_index < len(raw_data_full):
                break
            else:
                print(f"無効な開始インデックスです。0から{len(raw_data_full)-1}の間の整数を入力してください。")
        except ValueError:
            print("整数を入力してください。")

    while True:
        try:
            end_index = int(input(f"終了インデックス を入力してください ({start_index+1}から{len(raw_data_full)}の間): "))
            if start_index < end_index <= len(raw_data_full):
                break
            else:
                print(f"無効な終了インデックスです。開始インデックス({start_index})より大きく、{len(raw_data_full)}以下の整数を入力してください。")
        except ValueError:
            print("整数を入力してください。")
            
    raw_data_slice = raw_data_full[start_index:end_index]

    if len(raw_data_slice) == 0:
        raise ValueError("選択された時間範囲にデータがありません。")

    slice_label = f"slice_{start_index}-{end_index-1}"
    print(f"処理中のスライスラベル (自動生成): {slice_label}")

    data_min = np.min(raw_data_slice)
    data_max = np.max(raw_data_slice)

    if (data_max - data_min) == 0:
        print("警告: 選択されたデータ範囲の最小値と最大値が同じです。正規化を行いませんでした。")
        normalized_data = raw_data_slice 
    else:
        normalized_data = (raw_data_slice - data_min) / (data_max - data_min)

    print(f"\n--- データ前処理結果 (正規化) ---")
    print(f"解析対象データ範囲: インデックス {start_index} から {end_index-1} (データ点数: {len(normalized_data)})")
    print(f"正規化前のデータ最小値 (スライス範囲): {np.min(raw_data_slice):.3f}, 最大値: {np.max(raw_data_slice):.3f}")
    print(f"正規化後データ最小値: {np.min(normalized_data):.3f}, 最大値: {np.max(normalized_data):.3f}")

    plt.figure(figsize=(15, 5))
    plt.plot(normalized_data)
    plt.title(f'Normalized Data Slice (Index {start_index} to {end_index-1})')
    plt.xlabel('Data Point Index'); plt.ylabel('Normalized Value'); plt.grid(True)
    norm_data_buffer = io.BytesIO()
    plt.savefig(norm_data_buffer, format='png'); norm_data_buffer.seek(0)
    norm_data_filename = f'{file_name_without_ext}_normalized_data_slice_{start_index}-{end_index-1}.png'
    images_to_zip.append((norm_data_buffer, norm_data_filename))
    print(f"正規化データスライスのプロットをZIPファイルに含めます: {norm_data_filename}")
    plt.close()

except FileNotFoundError: print(f"エラー: ファイル '{file_path}' が見つかりません。"); exit()
except ValueError as e: print(f"エラー: {e}"); exit()
except Exception as e: print(f"予期せぬエラー: {e}"); exit()

# --- 3. 埋め込みパラメーターの設定と計算 ---
while True:
    try:
        embedding_dim = int(input(f"\n使用する埋め込み次元 (m, 2以上の整数) を入力してください: "))
        if embedding_dim >= 2: break
        else: print("2以上の整数を入力してください。")
    except ValueError: print("整数を入力してください。")
print(f"埋め込み次元 (m): {embedding_dim}")

max_lag_for_autocorr = len(normalized_data) // 4
if max_lag_for_autocorr == 0: max_lag_for_autocorr = 1; print("警告: データ範囲短すぎ。最大遅延時間=1。")
autocorr_values = calculate_autocorrelation(normalized_data, max_lag_for_autocorr)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(autocorr_values) + 1), autocorr_values, marker='o', markersize=4, linestyle='-')
plt.axhline(0, color='red', linestyle='--', label='Zero Autocorrelation')
plt.title(f'Autocorrelation Function for {file_name_without_ext} (Slice: {start_index}-{end_index-1})')
plt.xlabel('Lag'); plt.ylabel('Autocorrelation Coefficient'); plt.grid(True)
if max_lag_for_autocorr > 0: plt.xlim(0, max_lag_for_autocorr + 5)
else: plt.xlim(0, 5)
plt.legend(); autocorr_buffer = io.BytesIO()
plt.savefig(autocorr_buffer, format='png'); autocorr_buffer.seek(0)
autocorr_filename = f'{file_name_without_ext}_autocorrelation_plot_slice_{start_index}-{end_index-1}.png'
images_to_zip.append((autocorr_buffer, autocorr_filename))
print(f"自己相関プロットをZIPファイルに含めます: {autocorr_filename}")
plt.close()

optimal_lag = next((i + 1 for i, v in enumerate(autocorr_values) if v <= 0), max_lag_for_autocorr if max_lag_for_autocorr > 0 else 1)
time_delay = optimal_lag
print(f"遅延時間 (tau, 自己相関に基づき決定): {time_delay}")

# --- 4. メイン処理の実行 (リカレンスプロット生成 - Radiusを0~1の範囲で設定) ---
print(f"\n--- リカレンスプロット生成 (Radiusを0~1の範囲で指定) ---")
radius_values = np.arange(0.0, 1.0 + 1e-9, 0.01)
print(f"試行Radius値 (範囲 0.00-1.00, 0.01刻み, {len(radius_values)}個):")
if len(radius_values) < 20: print(radius_values)

total_start_time = time.time()
rr_values_for_plot = [] # RR vs Radius プロット用
radius_for_rr_plot = [] # RR vs Radius プロット用

if (data_max - data_min) == 0 and len(normalized_data) > 0:
    print("警告: 正規化データが全て同じ値のため、リカレンスプロット/RQA指標の生成をスキップします。")
else:
    for current_radius in radius_values:
        print(f"Radius = {current_radius:.3f} でRQA/RPを生成中...")
        time_series_data = normalized_data.tolist()
        if not time_series_data:
            print(f"  警告: Radius={current_radius:.3f} スキップ。正規化データが空。")
            continue

        time_series_obj = TimeSeries(time_series_data, embedding_dimension=embedding_dim, time_delay=time_delay)
        settings = Settings(time_series_obj, analysis_type=Classic, neighbourhood=FixedRadius(current_radius), similarity_measure=EuclideanMetric, theiler_corrector=1)

        rqa_metrics_for_current_radius = {
            'File': file_name_without_ext,
            'Slice_Start': start_index,
            'Slice_End': end_index,
            'Slice_Label': slice_label, 
            'Embedding_Dim': embedding_dim,
            'Time_Delay': time_delay,
            'Radius': current_radius
        }

        try:
            computation_rqa = RQAComputation.create(settings, verbose=False)
            result_rqa = computation_rqa.run()
            
            rqa_metrics_for_current_radius['RR_percent'] = result_rqa.recurrence_rate * 100 if result_rqa.recurrence_rate is not None else np.nan
            rqa_metrics_for_current_radius['DET_percent'] = result_rqa.determinism * 100 if result_rqa.determinism is not None else np.nan
            rqa_metrics_for_current_radius['LAM_percent'] = result_rqa.laminarity * 100 if result_rqa.laminarity is not None else np.nan
            rqa_metrics_for_current_radius['LMAX'] = result_rqa.longest_diagonal_line if result_rqa.longest_diagonal_line is not None else np.nan
            rqa_metrics_for_current_radius['ENTR_diag'] = result_rqa.entropy_diagonal_lines if result_rqa.entropy_diagonal_lines is not None else np.nan
            rqa_metrics_for_current_radius['AVG_diag_len'] = result_rqa.average_diagonal_line if result_rqa.average_diagonal_line is not None else np.nan
            rqa_metrics_for_current_radius['TrappingTime'] = result_rqa.trapping_time if result_rqa.trapping_time is not None else np.nan
            rqa_metrics_for_current_radius['VMAX'] = result_rqa.longest_vertical_line if result_rqa.longest_vertical_line is not None else np.nan
            rqa_metrics_for_current_radius['ENTR_vert'] = result_rqa.entropy_vertical_lines if result_rqa.entropy_vertical_lines is not None else np.nan
            
            rr_percentage = rqa_metrics_for_current_radius['RR_percent']

        except Exception as e_rqa:
            print(f"  警告: Radius={current_radius:.3f} でRQA計算中にエラー: {e_rqa}")
            keys_to_nan = ['RR_percent', 'DET_percent', 'LAM_percent', 'LMAX', 'ENTR_diag', 'AVG_diag_len', 'TrappingTime', 'VMAX', 'ENTR_vert']
            for key in keys_to_nan: rqa_metrics_for_current_radius[key] = np.nan
            rr_percentage = np.nan
        
        all_rqa_metrics_data.append(rqa_metrics_for_current_radius)
        print(f"    RR: {rr_percentage:.2f}%, DET: {rqa_metrics_for_current_radius.get('DET_percent', np.nan):.2f}%")
        
        if not np.isnan(rr_percentage):
            rr_values_for_plot.append(rr_percentage)
            radius_for_rr_plot.append(current_radius)

        try:
            computation_rp = RPComputation.create(settings)
            result_rp = computation_rp.run()
            recurrence_matrix = result_rp.recurrence_matrix
        except Exception as e_rp:
            print(f"  警告: Radius={current_radius:.3f} でRP計算中にエラー: {e_rp}"); plt.close(); continue

        plt.figure(figsize=(8, 8))
        if recurrence_matrix is not None and recurrence_matrix.size > 0:
            plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
            data_length_for_rp = recurrence_matrix.shape[0]
            if data_length_for_rp > 1: plt.xlim(0, data_length_for_rp - 1); plt.ylim(0, data_length_for_rp - 1)
            elif data_length_for_rp == 1: plt.xlim(-0.5, 0.5); plt.ylim(-0.5, 0.5)
        else:
            print(f"  情報: Radius={current_radius:.3f} RP行列空。プロット空白。"); plt.xlim(0, 1); plt.ylim(0, 1)
        plt.title(f'Recurrence Plot ({file_name_without_ext}, Slice: {start_index}-{end_index-1})\n(dim={embedding_dim}, delay={time_delay}, Radius={current_radius:.3f})\nRecurrence Rate (RR): {rr_percentage:.2f}%')
        plt.xlabel('Time (data points)'); plt.ylabel('Time (data points)'); plt.tight_layout()
        rr_for_filename = 'NaN' if np.isnan(rr_percentage) else f'{rr_percentage:.0f}'
        rp_filename = f'{file_name_without_ext}_rp_slice_{start_index}-{end_index-1}_dim{embedding_dim}_delay{time_delay}_Radius_{current_radius:.3f}_RR_{rr_for_filename}pc.png'
        rp_buffer = io.BytesIO(); plt.savefig(rp_buffer, format='png'); rp_buffer.seek(0)
        images_to_zip.append((rp_buffer, rp_filename))
        print(f"リカレンスプロットをZIPに含めます: {rp_filename}")
        plt.close()

# --- RRの推移プロット生成 (ZIP用) ---
if len(rr_values_for_plot) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(radius_for_rr_plot, rr_values_for_plot, marker='o', linestyle='-')
    plt.title(f'Recurrence Rate (RR) vs. Radius for {file_name_without_ext} (Slice: {start_index}-{end_index-1})')
    plt.xlabel('Radius'); plt.ylabel('Recurrence Rate (%)'); plt.grid(True)
    if len(radius_for_rr_plot) > 0: plt.xlim(np.min(radius_for_rr_plot), np.max(radius_for_rr_plot))
    plt.ylim(0, 100); rr_trend_buffer = io.BytesIO()
    rr_trend_filename = f'{file_name_without_ext}_rr_trend_slice_{start_index}-{end_index-1}.png'
    plt.savefig(rr_trend_buffer, format='png'); rr_trend_buffer.seek(0)
    images_to_zip.append((rr_trend_buffer, rr_trend_filename))
    print(f"RRの推移プロットをZIPファイルに含めます: {rr_trend_filename}")
    plt.close()
else:
    # ★変更点: RRトレンドの個別Excel出力は削除されたので、関連メッセージも調整
    print("警告: 有効なRR計算なし。RR推移プロットは生成されませんでした。")


# --- 全RQA指標データを1つのExcelファイルに保存/追記 ---
if all_rqa_metrics_data:
    slice_identifier_for_ml_filename = f"slice_{start_index}-{end_index-1}"
    ml_data_output_path = os.path.join(base_dir, f"{file_name_without_ext}_rqa_metrics_for_ml_{slice_identifier_for_ml_filename}.xlsx")
    
    new_ml_data_df = pd.DataFrame(all_rqa_metrics_data)

    if os.path.exists(ml_data_output_path):
        try:
            existing_ml_data_df = pd.read_excel(ml_data_output_path)
            combined_ml_data_df = pd.concat([existing_ml_data_df, new_ml_data_df], ignore_index=True)
            combined_ml_data_df.drop_duplicates(subset=['File', 'Slice_Start', 'Slice_End', 'Slice_Label', 
                                                         'Embedding_Dim', 'Time_Delay', 'Radius'], 
                                                keep='last', inplace=True)
        except FileNotFoundError:
             print(f"情報: 既存の機械学習用データファイル {ml_data_output_path} が見つかりませんでしたが、新規作成します。")
             combined_ml_data_df = new_ml_data_df
        except Exception as e_read_excel:
            print(f"警告: 既存の機械学習用データファイル {ml_data_output_path} の読み込み中にエラー: {e_read_excel}。")
            print(f"新しいデータでファイルを作成しなおします（既存の同名ファイルの内容は失われます）。")
            combined_ml_data_df = new_ml_data_df 
    else:
        combined_ml_data_df = new_ml_data_df
    
    try:
        combined_ml_data_df.to_excel(ml_data_output_path, index=False)
        print(f"全RQA指標データを機械学習用Excelファイルに保存/追記しました: {ml_data_output_path}")
        print(f"  (注意: Excel書き込みには 'openpyxl' が必要です。 pip install openpyxl )")
    except Exception as e:
        print(f"  警告: 全RQA指標データのExcelファイル保存中にエラーが発生しました: {e}")
else:
    print("収集されたRQA指標データがありません。機械学習用ファイルは作成/更新されませんでした。")


total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"\n--- 全解析終了 ---")
print(f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

if images_to_zip:
    zip_file_name = os.path.join(base_dir, f'{file_name_without_ext}_rp_analysis_results_slice_{start_index}-{end_index-1}.zip')
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_buffer, img_filename in images_to_zip:
            zipf.writestr(img_filename, img_buffer.getvalue())
            print(f"'{img_filename}' をZIPファイルに追加しました。")
    print(f"\n--- 全ての画像がZIPファイルにまとめられました: {zip_file_name} ---")
else:
    print("\n--- ZIPファイルにまとめる画像がありませんでした。 ---")