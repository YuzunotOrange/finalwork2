#include<stdio.h>
#include<math.h>
typedef struct 
{
  double   x,y,z;
}VECTOR;

double dot(VECTOR a, VECTOR b);
VECTOR cross(VECTOR a,VECTOR b);
int main(){
    VECTOR a,b,c;
    printf("一番目のベクトルaのｘ、ｙ、ｚ成分を空白を用いて入力してください→");
    scanf("%lf%lf%lf",&a.x,&a.y,&a.z);
    printf("二番目のベクトルｂのｘ、ｙ、ｚ成分を空白を用いて入力してください→");
    scanf("%lf%lf%lf",&b.x,&b.y,&b.z);
    printf("Answer→%f\n",dot(a,b));
    return 0;
}
double dot(VECTOR a,VECTOR b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
