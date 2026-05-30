//1-12
//2022/9/27
//bp22104 松本　優瑞
#include<stdio.h>
int main(void)
{
    int a,b,temp;
    scanf("%d,%d", &a,&b);
    temp = a;
    a = b;
    b = temp;
    printf("%d,%d",a,temp);
    return 0;
}

