#include<stdio.h>
#include<stdlib.h>
int main(void){
    int *p;
    int i;
    p=(int *)malloc(sizeof(int)*5);
    scanf("%d\n",p);
    
free(p);
return 0;
}