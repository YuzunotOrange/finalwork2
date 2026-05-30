#include<stdio.h>
#include<stdlib.h>

int main(void){
    float *p;
    /*メモリ領域を動的確保しアドレスを変数ｐに格納*/
    p=(float *)malloc(sizeof(float));
    scanf("%f",p);
    if(*p > 0)
    printf("%f\n",*p);
    else
    printf("%f\n",-(*p));
    //動的確保したメモリー領域を解放
    fraa(p);

return 0;
}