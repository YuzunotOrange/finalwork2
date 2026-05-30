#include<stdio.h>
int main(void)
{
    int x = 1, y = 0;
    for(x<=10;y=y+x;x=x+1);
    printf("%d\n", y);
    return 0;
}