#include <stdio.h>
#include <string.h>
#include <openssl/des.h>

// DES暗号化と復号化の関数
void des_encrypt(DES_cblock *key, DES_cblock *input, DES_cblock *output) {
    DES_key_schedule schedule;
    DES_set_key_checked(key, &schedule);
    DES_ecb_encrypt(input, output, &schedule, DES_ENCRYPT);
}

void des_decrypt(DES_cblock *key, DES_cblock *input, DES_cblock *output) {
    DES_key_schedule schedule;
    DES_set_key_checked(key, &schedule);
    DES_ecb_encrypt(input, output, &schedule, DES_DECRYPT);
}

int main() {
    // 8バイトのキーと初期化ベクトル
    DES_cblock key = {0x12, 0x34, 0x56, 0x78, 0x90, 0xAB, 0xCD, 0xEF};
    DES_cblock input = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    DES_cblock output;
    DES_cblock decrypted;

    printf("Original Text: ");
    for (int i = 0; i < 8; i++) {
        printf("%02X ", input[i]);
    }
    printf("\n");

    // 暗号化
    des_encrypt(&key, &input, &output);
    printf("Encrypted Text: ");
    for (int i = 0; i < 8; i++) {
        printf("%02X ", output[i]);
    }
    printf("\n");

    // 復号化
    des_decrypt(&key, &output, &decrypted);
    printf("Decrypted Text: ");
    for (int i = 0; i < 8; i++) {
        printf("%02X ", decrypted[i]);
    }
    printf("\n");

    return 0;
}
