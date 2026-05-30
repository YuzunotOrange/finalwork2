#include<stdio.h>
int main(void){
    float f, *p;
    p = &f;
    scanf("%f",p);
    printf("f = %f, &f = %p\n",f, &f);
    printf("p = %p, *p = %f\n",p, *p);
    printf("&p = %p\n", &p);
    return 0;
}