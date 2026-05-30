#include<stdio.h>
#include<stdlib.h>
int main(void)
{
    int i, *array;
    array = (int*)malloc(sizeof(int)*5);//メモリー領域に各要素を連続して格納している
    for(i=0; i<5; i++)
    scanf("%d",array+i);//array+iの意味
    for (i=0; i<5; i++)
    printf("arrry[%d] = %d\n", i, array[i]);//array[i]の意味
    free(array);//メモリー領域を解放する
    return 0;
    
}