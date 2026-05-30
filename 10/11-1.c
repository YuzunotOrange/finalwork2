#include<stdio.h>
#include<stdlib.h>
int main(void){
    int *p;
    while (1){
    
     p=(int *)malloc(sizeof(int));
     printf("数値を入力してください→");
     scanf("%d",p);
     if(*p<0){
        break;
     }
    }
     printf("%d,%p\n",*p,p);
     free(p);
     return 0;
}