#include<stdio.h>
struct private_info
{
    /* data */
    char id[10], name[100], highschool[100];
    int year, month;
};
int main(void)
{
    
    struct private_info my_data[3], *p;
    p = my_data;
    printf("学籍番号:");
    scanf("%s",p->id);
    printf("氏名:");
    scanf("%s",p->name);
    printf("出身高校:");
    scanf("%s",p->highschool);
    printf("生まれた年:");
    scanf("%d",&p->year);
    printf("生まれた月:");
    scanf("%d",&p->month);

    printf("学籍番号:%s\n",p->id);
    printf("氏名:%s\n",p->name);
    printf("出身高校:%s\n",p->highschool);
    printf("生まれた年:%d\n",p->year);
    printf("生まれた月:%d\n",p->month);
}
