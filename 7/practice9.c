#include<stdio.h>
#include<stdlib.h>
//構造体の型定義
struct score
{
    char id[8];
    char name[16];
    int math;
    int physics;
    int english;
    int average;
};
int main(void){
    //構造体の変数定義
    struct  score x;
    //　各メンバへの代入や参照など
    //(ここではidとnameが配列であることに注意)
    scanf("%s", x.id);
    scanf("%s", x.name);
    scanf("%d", &x.math);
    scanf("%d", &x.physics);
    scanf("%d", &x.english);
    x.average = (x.math + x.physics + x.english)/3.0;
    //各メンバの表示例
    printf("ID:%s,NAME:%s\n",x.id ,x.name);
    printf("math:%d, physics:%d, English:%d\n",x.math ,x.physics ,x.english);
    printf("average:%f\n",x.average);
    return 0;
    
}