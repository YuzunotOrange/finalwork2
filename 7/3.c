#include<stdio.h>
int square_sum(int);
int main(void){
    int n;
    scanf("%d",&n);
    printf("%d\n",square_sum(n));
    return 0;
}
int square_sum(int n){
    if (n == 1) return 1;
    else return n*n+square_sum(n-1);
}