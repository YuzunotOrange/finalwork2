#include <stdio.h>
 void clac( int *x, int *y){
    int z;
    z = *x;
    *x = *y;
    *y = z;
}
int main(void){
    int  a = 5, b = 3;
    clac(&a, &b);
    printf("a = %d\n", a );
    printf("b = %d\n", b);
    return 0;
}