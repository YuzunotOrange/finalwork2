#include <stdio.h>

int main(void)
{
int i, j;
float matrix[3][2];
for (i = 0; i < 3; i++) {
for (j = 0; j < 2; j++) {
printf("%d行%d列のデータ：", i + 1, j + 1);
scanf("%f", &matrix[i][j]);
}
}
for (i = 0; i < 3; i++) {
for (j = 0; j < 2; j++) {
printf("matrix[%d][%d] = %f\n", i + 1, j + 1, matrix[i][j]);
}
}
return 0;
}