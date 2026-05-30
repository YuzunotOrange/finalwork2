#include <stdio.h>
int main(){
    int a = 1;
    int *b = &a;
    int d = a + a;  
    int e = *b;
    a += 2;
    printf("%d\n", d);
    printf("%d\n", e);
 return 0;

}