#include <stdio.h>

int main(void)
{
int i, j;
float A[2][2];
for (i = 0; i < 2; i++) {
for (j = 0; j < 2; j++) {
printf("%d行%d列のデータ：", i + 1, j + 1);
scanf("%f", &A[i][j]);
}
}
for (i = 0; i < 2; i++) {
for (j = 0; j < 2; j++) {
printf("matrix[%d][%d] = %f\n", i + 1, j + 1, A[i][j]);
}
}
return 0;
}