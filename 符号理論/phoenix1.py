import random
import string

#定数を定義する部分
ALL_CHARS = string .ascii_uppercase + string.digits + string.punctuation + string.whitespace #アルファベットや数字、句読点や記号を定義
MAX_SYMBOLS = 5 #最大のシンボルの数
SYMBOL_RANGE = 1000 #各アルファベット文字に対応するランダムなシンボルの範囲

#各アルファベットに対してランダムなシンボルを作成する
def generate_random_symbols():
    symbols = {}
    for char in ALL_CHARS:#アルファベットや数字、句読点や記号もランダムなシンボルを作成する使用
        symbols[char] = [random.randint(1,SYMBOL_RANGE) for _ in range(MAX_SYMBOLS)]
    return symbols

#テキストを暗号化する関数
def encrypt(plaintext, symbols):
    #ciphertextという空のリストを作成
    ciphertext = []
    for char in plaintext:
        if char in symbols:
            possible_symbols = symbols[char]
            ciphertext.append(random.choice(possible_symbols))
        else:
            ciphertext.append(char) #辞書にない文字はそのまま
    return ciphertext


def main():
    symbols = generate_random_symbols()
    #暗号化する文章（平文）
    text = input("文章を入力してください:")
    print(f"入力された文章: {text}")

    #暗号化
    encrypted = encrypt(text, symbols)
    print(f"暗号化された文章: {''.join(map(str, encrypted))}")

if __name__ == "__main__":
    main()