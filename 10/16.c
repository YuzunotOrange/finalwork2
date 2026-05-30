#include<stdio.h>
int main(void)
{
    float sum=0.0,data,num=0.0;
    int i;
    while(1){
        printf("データを入力してください＞");
        scanf("%f",&data);
        if(data<0.0)break;
        sum=sum+data;
        num=num+1.0;
         }
printf("%f",sum/sum);
return 0;
}