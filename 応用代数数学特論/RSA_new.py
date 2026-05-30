# --- 1. 与えられた公開鍵と素因数分解のヒント ---
e = 3
p = 131
q = 503
n = p * q  # 65893

# --- 2. 秘密鍵 d の計算 ---
phi = (p - 1) * (q - 1)  # 65260
d = pow(e, -1, phi)
print(f"算出された秘密鍵 d: {d}")

# --- 3. 解読する暗号文のリスト ---
cipher_text = [
    30419,
    53161,
    59465,
    30093,
    46474,
    63534,
    57921,
    41357,
    53248,
    18766,
]

# --- 4. 復号処理 ---
decrypted_message = ""

# 【修正ポイント】問題文の対応表（0番目＝空白、1番目＝A、...）をそのまま定義
chars_table = {i:c for i,c in enumerate(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

for c in cipher_text:
    # ステップ A: c^d mod n を計算
    m = pow(c, d, n)

    # ステップ B: 100で割った余り（x）を出す
    x = m % 100

    # ステップ C: 【修正】対応表から直接文字を引っ張ってくる（安全で確実）
    # もし計算ミス等で x が 27 以上になってしまった場合のバグを防ぐガードも兼ねています
    if 0 <= x < len(chars_table):
        char = chars_table[x]
    else:
        char = "?"  # 想定外の数値のときはハテナにする

    decrypted_message += char

# --- 5. 結果の表示 ---
print(f"解読結果: {decrypted_message}")
print(c, m, x, char)