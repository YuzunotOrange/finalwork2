#include<stdio.h>
int sub(int);
int main (void){
    int a,b;
    a = 2;
    b = sub(a)+1;
    printf("%d\n",b);
    return  0;
}
int sub(int x){
    int b,z;
    b = 3;
    z = 2*x;
    return z;
}