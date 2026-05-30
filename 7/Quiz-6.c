#include<stdio.h>
int main(void){
int i, j;
int kuku[9][9];
i=0; j=0;
while(i<9){
while(j<9){
kuku[i][j]=(i+1)*(j+1);
if(j+1==9){
    j=0;
break;
}
j++;
}
i++;
}
for(i=0; i<9; i++){
for(j=0; j<9; j++){
printf("%5d", kuku[i][j]);
}
printf("\n");
}
return 1;
}