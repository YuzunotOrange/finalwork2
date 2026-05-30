#include<stdio.h>
#include<stdlib.h>

    void max_and_min (const int array[], int num, int *max, int *min)
{
  int i;
  *max = *min = array[0];

    for (i=0; i<num; i++) {
      if (array[i] > *max) 
	*max = array[i];
      if (array[i] < *min) 
	*min = array[i];
    }
  }


int main(void)
{
  int i, n, array[n],max, min;
 do {
    printf("値を入力してください→");
    scanf("%d\n", &n);
    for(int i=0; i<n; i++);
    scanf("%d",&array[i]);
    i++;
 } while (n != 0);

  max_and_min(array, n, &max, &min);

  printf("%d %d\n", max, min);

  return 0;
}