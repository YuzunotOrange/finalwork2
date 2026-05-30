#include <stdio.h>


int count;
int f(int n){
count++;
if (n <= 1) {
return 1;
} else if (n == 2) {
return 2;
} else {
return n * f(n - 1);
}
}

int main() 
{
for (int i = 0;i < 10;i++) {
count = 0;
printf("f(%d) = %d\n",i,f(i));
printf("count = %d\n\n",count);
}
return 0;
}