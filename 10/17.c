#include <stdio.h> 
int main(void){
char data; 
printf("アルファベットを入力して下さい＞");
scanf("%c", &data);
printf("%c is %d-th alphabet\n", data, data-'A'+1); 
return 0;
}