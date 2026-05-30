//2-9
//2022/10/4
//bp22104 松本　優瑞
#include<stdio.h>
int main(void)
{
    char data;
    printf("アルファベットを入力：");
    scanf("%c",&data);
    if(97 <= data && data <= 122){
    data = data - 32;
    printf("対応する大文字は%cです\n",data);}
    else if(65 <= data && data <= 90){
    data = data + 32;
    printf("対応する小文字は%cです\n",data);
    }
    else
    {
        printf("error");
    }

    return 0;
}