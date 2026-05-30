#include<stdio.h>

void func(int[][9]);
void show(int[][9]);

int main() {
int kuku[9][9];
func(kuku);
show(kuku);
return 0;
}

void func(int kuku[][9]) {
for (int i = 0; i < 9; i++) {
for (int j = 0; j < 9; j++) {
kuku[i][j] = (i+1) * (j+1);
}
}
}

void show(int kuku[][9]) {
for (int i = 0; i < 9; i++) {
for (int j = 0; j < 9; j++) {
printf("%2d, ", kuku[i][j]);
}
printf("\n");
}
}