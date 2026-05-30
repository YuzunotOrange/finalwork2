/* 情報処理演習2
   課題5-4
   2022年10月31日
   BP22104 松本　優瑞
*/

#include<stdio.h>
#include<math.h>
int main(void) {
    int i;
    double tmp,data[10],datasub[10],sum=0.0,maximum,minimum,average=0.0,variance=0.0,standarddeviation=0.0;
  
    for(i=0;i<10;i++) {
        printf("データを入力してください:");
        scanf("%lf",&data[i]);
        if(i==0){
            maximum=data[0];
            minimum=data[0];
        }
        if(maximum<data[i]) {
            maximum=data[i];
        }
         if(minimum>data[i]) {
            minimum=data[i];
        }
        sum=sum+data[i];
    }
    average=sum/10;
    for (i = 0; i < 10; i++) {
        variance=variance+(data[i]-average)*(data[i]-average);
    }
    standarddeviation=sqrt(variance);
    for(i=0;i<10;i++) {
        datasub[i]=data[9-i];
    }
    for(i=9;i>=0;i--) {
        tmp=datasub[i];
        data[i]=tmp;
    }
    for(i=0;i<10;i++) {
        printf("revers:%f\n",data[i]);
    }
    for(i=0;i<10;i++) {
        datasub[i]=data[9-i];
    }
    for(i=0;i<10;i++) {
        tmp=datasub[i];
        datasub[i]=tmp;
    }

    printf("最大=%f,最小=%f,平均=%f,分散=%f,標準偏差=%f\n",maximum,minimum,average,variance,standarddeviation);
    return 1;
    
}