#include<stdio.h>
#include<stdlib.h>
//構造体の型定義
struct Movie{
    int year;
    char title[50];
    int W_money;
   
};

int main(void){
    //構造体配列の定義
    struct Movie x[3];
    int i;
    for(i=1; i<3; i++){
    printf("rank[%d]\n",i);
    printf("yaer=");
    scanf("%d",&x[i].year);
    printf("titele=");
    scanf("%s",x[i].title);
    printf("World_Box_Office=");
    scanf("%d",&x[i].W_money);
    }
for(i=0; i<20; i++);{
    printf("rank[%d]\n",i);
    printf("Year:%d\n",x[i].year);
    printf("Title:%s\n",x[i].title);
    printf("World_Box_Office:%d\n",x[i].W_money);
    }
    return 0;
}
