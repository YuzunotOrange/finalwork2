import os

def generate_key(length):
    #指定された長さのランダムなキーを作成する
    return os.urandom(length)

#平文とキーを使用して暗号化する
def encrypt(plaintext, key):
    #暗号化するために文字列をバイト列に変換する
    plaintext_bytes = plaintext.encode('utf-8')
    #平文のバイト列とキーのバイト列をXOR演算
    ciphertext = bytes([p ^ k for p, k in zip(plaintext_bytes, key)])
    return ciphertext

#暗号文とキーを使って復号化
def decrypt(ciphertext, key):
    #暗号のバイト列とキーのバイト列のXOR演算をして復号化
    decrypted_bytes = bytes([c ^ k for c, k in zip(ciphertext, key)])
    return decrypted_bytes.decode('utf-8')

# 平文の入力
plaintext = "I am YUZU MATSUMOTO student of SIT!!!"

# 平文と同じ長さのランダムなキーの作成
key = generate_key(len(plaintext))

# 暗号化
ciphertext = encrypt(plaintext, key)
print(f"暗号文: {ciphertext}")

# 復号化
decrypted_text = decrypt(ciphertext, key)
print(f"復号文: {decrypted_text}")
