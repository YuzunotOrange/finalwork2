//1-13
//2022/9/27
//bp22104 松本　優瑞
#include<stdio.h>
int main(void)
{
    //整数型変数 x，y，z を定義する
    int x,y,z;
    //x と y にキーボードから値を入力する．
    scanf("%d,%d", &x,&y);
    // x＋y の値を z に代入する
    z = x + y;
    //計算式と計算結果を画面表示する
    printf("%d = %d + %d\n", z, x, y);
    //x－y の値を z に代入する
    z = x - y;
    //計算式と計算結果を画面表示する
    printf("%d = %d - %d\n", z, x, y);
    return 0;
}
    
