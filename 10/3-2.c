#include <stdio.h>
int sq(int);
int main( void )
{
    int x,y;
    scanf("%d,%d",&x,&y);
    printf("%d\n",sq(x)+sq(y));
    return 0;

}
int sq(int n)
{
    return n*n;
}