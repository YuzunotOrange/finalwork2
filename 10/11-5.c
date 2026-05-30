/*11-5.c
  2022-12-16
  BP22104
  松本　優瑞*/
#include<stdio.h>
#include<stdlib.h>
int main(void){
    int n,m;
    int i,j;
    int **array;
    printf("m=");
    scanf("%d",&m);
    printf("n=");
    scanf("%d",&n);
    array=(int**)malloc(sizeof(int*)*m);
    for(i=0;i<m;i++);{
        array[i]=(int*)malloc(sizeof(int*)*n);
    }
    for(i=0;i<m;i++){
    for(i=0;i<m;i++){
        array[i][j]=0;
    }    
    }
    for(i=0;i<m;i++){
        printf("array");
free(array);
    }


}