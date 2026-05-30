/*8-8
  2022-11-28
  BP22104
  松本　優瑞*/
#include <stdio.h>
int main( void )
{
 int i;
 int score[5];
 char name[5][64];
 int max_i;
 double ave = 0;
 /*氏名と点数の入力 */
 for ( i=0; i<5; i++ ) {
 printf( "%d 番目の名前と点数を入力して下さい： ", i );
 scanf("%s %d", &name[i],&score[i]);
 }
 /*平均点計算*/
 for (i=0; i<5; i++)
 ave += score[i];
 ave /= 5.0;
 /*最高点計算*/
 int max;
 max = 0;
 for(i=1; i<6; ++i)
 {
    if(max<score[i]){
        max=score[i];
        max=i;}
    }
    
 
/* 画面表示 */
 printf( "平均点： %f\n", ave );
 printf( "最高点： %s %d\n",name[max], score[max] );
 return 0;
}

