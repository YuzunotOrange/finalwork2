#include<stdio.h>
struct vector{
    double x, y;
};
int main(void){
    struct vector v1, v2;
    scanf("%lf", &v1.x);
    scanf("%lf", &v1.y);
    v2.x = -v1.x;
    v2.y = -v1.y;
    printf("v1 = (%f,%f)\n",v1.x ,v1.y);
    printf("v2 = (%f,%f)\n",v2.x ,v2.y);
    return 0;
}