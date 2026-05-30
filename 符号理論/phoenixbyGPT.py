import random

# ホモフェニック暗号のパラメータ
p = 257  # 素数（mod pで計算する）
g = 2    # 生成元

# 鍵生成
def generate_key():
    sk = random.randint(1, p-1)  # 秘密鍵
    pk = pow(g, sk, p)           # 公開鍵
    return sk, pk

# 暗号化
def encrypt(pk, m):
    r = random.randint(1, p-1)
    c1 = pow(g, r, p)
    c2 = (m * pow(pk, r, p)) % p
    return c1, c2

# 復号化
def decrypt(sk, c1, c2):
    s_inv = pow(c1, sk, p)       # c1^sk
    s_inv = pow(s_inv, p-2, p)   # c1^(-sk) (mod p)
    m = (c2 * s_inv) % p
    return m

# メイン関数
def main():
    # 鍵の生成
    sk, pk = generate_key()
    print("秘密鍵:", sk)
    print("公開鍵:", pk)

    # 平文の設定
    m = 123
    
    # 暗号化
    c1, c2 = encrypt(pk, m)
    print("暗号文 c1:", c1)
    print("暗号文 c2:", c2)

    # 復号化
    decrypted_message = decrypt(sk, c1, c2)
    print("復号化された平文:", decrypted_message)

if __name__ == "__main__":
    main()
