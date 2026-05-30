#include<stdio.h>
void value_change( int *, int *);
int main(void)
{
    int a,b;
    printf("2つの整数を入力してください→");
    scanf("%d %d", &a, &b);
    printf("a = %d, b = %d\n", a, b);
    value_change(&a, &b);
    printf("a = %d, b = %d\n", a, b);
    return 0;
}
void value_change(int *a, int *b){
    int c;
    c = *a;
    *a = *b;
    *b = c;
    
}