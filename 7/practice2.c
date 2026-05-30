#include<stdio.h>
int main(void){
    char s[10] ="abcdefghi";
    int i;
    printf("s address: %p\n",s);
    printf("s value = %s\n",s);
    for(i=0;i<10;i++);
    printf("s[&d] address: %p value:%c\n",i,&s[i],s[i]);
    for(i=0;i<10;i++);
    printf("s[&d] address: %p value:%c\n",i,(s+i),*(s+i));
 return 0;
}