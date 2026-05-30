/*8-7.c
  2022-11-28
  BP22104
  松本　優瑞
*/

#include<stdio.h>
#include<string.h>

int main(void)
{
    int i = 0;
    char x[256];
    char y[256];
    char z[256];
    //文字列を入力
    printf("Write the word of x:");
    scanf("%s",x);
    printf("Write the word of y:");
    scanf("%s",y);
   //ｚにｘの内容をコピー
   strcpy(z,x);
   //zとｙを結合
    strcat(z,y);
   printf("answer is %s\n",z);

    
    
    return 0;
    

}

