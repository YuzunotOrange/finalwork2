//2-1
//2022/10/4
//bp22104 松本　優瑞
#include<stdio.h>
int main(void)
{
    int a, b, c;
    float d;
    printf("3つの整数を入力してください　→");
    scanf("%d %d %d", &a, &b, &c);
    d = (a + b + c) / 3;
    printf("入力された整数の平均は %f です\n", d);
    return 0;
}