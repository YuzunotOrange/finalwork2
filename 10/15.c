#include<stdio.h>
int main(void)
{
    float sum=0,data;
    int i,num;
    printf("個数を入力>");
   scanf("%d",&num);
   for(i=1;i<=num;i++){
    printf("データを入力>");
    scanf("%f",&data);
    sum = sum + data;
   }
   printf("%f\n",sum/sum);
   return 0;
}