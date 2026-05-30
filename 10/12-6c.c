/*12-6.c
2023-1-9
BP22104
松本優瑞*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
struct station_info {
char name[100];
double latitude;
double longitude;
};
int main()
{
struct station_info entry[] =
{
{"浦和駅" ,35.513,139.392 }
,{"さいたま新都心駅",35.533,139.380 }
,{"大宮駅" ,35.542,139.372 }
,{"土呂駅" ,35.555,139.375 }
,{"東大宮駅" ,35.565,139.381 }
};

printf("北緯と東経の値を入力してください。\n");
double lat,lon;
scanf("%lf%lf", &lat,&lon);

double kyori_min = 999.99;//この値は暫定値
int min_check = 0;
for (int i = 0; i < (sizeof entry / sizeof * entry); ++i) {
const double kyori = sqrt(pow((double)fabs(entry[i].latitude - lat ),2.0)
+pow((double)fabs(entry[i].longitude - lon ),2.0));
if (kyori_min < kyori)
continue;
else {
kyori_min = kyori;
min_check = i;
}
}
//駅名と緯度経度情報を表示
printf("駅名：%s\n",entry[min_check].name );
printf("緯度：%f\n",entry[min_check].latitude );
printf("経度：%f\n",entry[min_check].longitude);
return 0;
}