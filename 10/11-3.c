/*11-3.c
  2022-12-14
  BP22104
  松本　優瑞*/
#include<stdio.h>
#include<stdlib.h>
void input(int*array,int n){
    for(int i=0; i<n; i++){
    printf("array[%d] > ",i);
    scanf("%d",&array[i]);        
}
}
int max(int*array, int n){
    int max=array[0];
    for(int i=1; i<n; i++)
        if(array[i]>max)max=array[i];
        
        return max;
        
}
int min(int*array,int n){
    int min=array[0];
    for(int i=1; i<n; i++)
        if(array[i]<min)min=array[i];
        return min;
    

}

double avg(int *array, int n ){
    int sum=0;
    for(int i=1;i<n;i++)
    sum += array[i];
    
    return (double)sum/n;
}

int main(void){
    int n;
    int *a;
    printf("n?>");
    scanf("%d",&n);
    a=(int*)malloc(sizeof(int)*n);
    input(a,n);
    printf("max: %d\n",max(a,n));
    printf("min: %d\n",min(a,n));
    printf("avg: %f\n",avg(a,n));
    free(a);
    return 0;

}
        
