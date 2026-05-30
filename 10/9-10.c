/*9-10
  2022-11-29
  bp22104
  松本　優瑞*/
#include<stdio.h>
int main(void){
    char *p;
    char *q;
    char s[256];
    char t[256];
    printf("Write the word of S:");
    scanf("%s",s);
    p = s;
    q = t;
    while(*p != '\0'){
        *q = *p;
        p++;
        q++;
        }
        printf("文字列は%s",t);
        return 0;

}