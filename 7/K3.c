#include<stdio.h>
int fact(int);
int main(void){
int x;
scanf("%d", &x);
printf("%d!=%d\n", x, fact(x));
return 0;
}
int fact(int n){
if(n==1) return 1;
else return n*fact(n-1);

} 