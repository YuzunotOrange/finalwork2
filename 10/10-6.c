/*10-6.c
  2022-12-7
  BP22104
  松本優瑞*/
#include <stdio.h>
#include <math.h>

int solve( float a, float b, float *x1, float *x2);

int main(void){
    float a, b, x1, x2;
    int c;
    printf("ｘの係数：");
    scanf("%f", &a);
    printf("定数項ㅤ：");
    scanf("%f", &b);
    c = solve(a, b, &x1, &x2);
    switch (c){
    case 2:
        printf("実数解2つ\n");
        printf("x = %f\n", x1);
        printf("x = %f\n", x2);
        break;
    case 1:
        printf("重解\n");
        printf("x = %f\n", x1);
        break;
    case 0:
        printf("実数解なし\n");
        break;    
    default:
        break;
    }
}

int solve( float a, float b, float *x1, float *x2){
    float D;
    D = a * a - 4 * b;
    if(D < 0){
        return 0;
    }else if(D == 0){
        *x1 = (-a+sqrt(D))/2.0;
        return 1;
    }else{
        *x1 = (-a+sqrt(D))/2.0;
        *x2 = (-a-sqrt(D))/2.0;
        return 2;
    }
}