#include <stdio.h>
struct station_info {
 char name[100];
 double latitude;
 double longitude;
};
struct station_info input( void )
{
 struct station_info entry;
 printf("駅名を入力→ " );
 scanf("%s", entry.name );
 printf("北緯を入力→ " );
 scanf("%lf", &entry.latitude );
 printf("東経を入力→" );
 scanf("%lf", &entry.longitude );
 return entry;
}
int main( void )
{
 struct station_info station;
 station = input();
 printf( "駅名：%s, 北緯%.2f 度，東経%.2f 度\n",
station.name, station.latitude, station.longitude );
 return 0;
}
