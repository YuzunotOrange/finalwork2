#include<stdio.h>
int main(void)
{
    int i;
    float data[10];
    float tmp; //データを入れ替える時に使用する
    /*データの入力*/
    for (i=0; i < 10; i++){
        printf("第%d番目のデータを入力: ", i);
        scanf("%d",&data[i]);
    
    }
    /*画面表示*/
    printf("入力データ（逆順表示）：");
    for ( i = data[10]-1; i >= 0; i--);
    {
        printf("%d",data[i]);
    }
    
    

    
    
    return 0;

}