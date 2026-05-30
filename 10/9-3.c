#include<stdio.h>
int main(void){
    int a[5] = {3, 4, -5, 2, 10};
    int *p;
    int i;
    p = a;
    for (i=0; i<5; i++)
    printf("p+%d = %p\n", i, p+i);
    for (i=0; i<5; i++)
    printf("*(p+%d) = %d\n", i, *(p+i));
    return 0; 
}