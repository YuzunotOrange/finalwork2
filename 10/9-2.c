#include<stdio.h>
int main(void){
    int i;
    int a[5] = {3, 4, -5, 2, 10};
    for (i=0; i<5; i++){
    printf("a[%i]= %d,a[%i]のアドレス=%p\n",i, a[i],&a[i]);
    }
    return 0;
}