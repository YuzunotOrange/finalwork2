#include<stdio.h>
int recursive_kaijo ( int i )
{
    if (!i)return 1;
    else return i*recursive_kaijo(i-1);  

    
}
int main(void)
{
    int a;
    do
    {
        printf("0以上の整数値を入力してください→");
        scanf("%d", &a);
    } while (a < 0);
    printf("%dの階乗は%dです。\n", a, recursive_kaijo(a));
    return 0;
    
}
