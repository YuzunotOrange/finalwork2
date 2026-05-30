#include<stdio.h>
int function (int);
int main (void)
{
    int n;
    do{
        printf("0以上の整数値を入力してください→");
        scanf("%d", &n);
    }while (n < 0);
    printf("計算結果: %d\n", function (n));
    return 0;
    }
int function(int i){
    if (i == 1) return 1;
    else return i*function(1);
}