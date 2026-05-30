#include <stdio.h>

int main() {
    char input;
    
    // ユーザーから入力を受け取る
    printf("文字: ");
    scanf("%c", &input);

    // 入力が小文字のアルファベットかどうかを判定
    if (input >= 'a' && input <= 'z') {
        // アルファベットの何番目かを計算 ('a' が 1 番目)
        int position = input - 'a' + 1;
        printf(" %d 番目\n", position);
    } else {
        printf("小文字ではない。\n", input);
    }

    return 0;
}
