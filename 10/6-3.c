#include <stdio.h>

int main(void)
{
int i, j,c;
float A[2][2];
for (i = 0; i < 2; i++) {
for (j = 0; j < 2; j++) {
printf("%d行%d列のデータ：", i + 1, j + 1);
scanf("%f", &A[i][j]);
}
}
c = A[0][0] * A[1][1] - A[0][1] * A[1][0];
printf("|A| = %d\n",c);
return 0;
}
