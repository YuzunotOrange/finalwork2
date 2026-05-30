#include<stdio.h>
void max_min( int *, int *);
int main(void)
{
    int a,b;
    printf("2つの整数を入力してください→");
    scanf("%d %d", &a, &b);
    printf("a = %d, b = %d\n", a, b);
    max_min(&a, &b);
    printf("a = %d, b = %d\n", a, b);
    return 0;
}
void max_min(int *a, int *b){
    int c;
    if ( *b < *a){
        c = *b;
     *b = *a;
        *a = c;
    }
}