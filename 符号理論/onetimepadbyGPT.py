def one_time_pad_encrypt(plain_text, key):
    encrypted_text = []
    for i in range(len(plain_text)):
        encrypted_char = chr(ord(plain_text[i]) ^ ord(key[i]))
        encrypted_text.append(encrypted_char)
    return ''.join(encrypted_text)

def main():
    # 平文と鍵の準備
    plain_text = "HELLO"
    key = "SECRET"

    # 暗号化
    cipher_text = one_time_pad_encrypt(plain_text, key)

    # 暗号文を16進数で表示
    hex_cipher_text = ' '.join(format(ord(c), '02x') for c in cipher_text)

    print("平文:", plain_text)
    print("鍵:", key)
    print("暗号文:", hex_cipher_text)

if __name__ == "__main__":
    main()
