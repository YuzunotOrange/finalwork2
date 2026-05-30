#include<stdio.h>
int calc_abs( int i )
{
    
    int a;
    if (i > 0)
    return i;
    else
    return -i;
  a = i;
    
}

int main(void)
{
        int a, abs_a;
        printf("整数値を入力してください→");
        scanf("%d",&a); 
       abs_a = multiple(a);
    printf("%dの絶対値は%dです。\n",a, abs_a);
return 0;

}