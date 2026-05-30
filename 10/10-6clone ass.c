#include <stdio.h>
#include <math.h>

struct answer {
int num;
float ans1, ans2;
};

struct answer ans_ins;

struct answer solve(float a, float b) {
float det;

det = a * a - 4.0 * b;

if(det > 0) {ans_ins.num = 2;
ans_ins.ans1 = (-a + sqrt(det)) / 2;
ans_ins.ans2 = (-a - sqrt(det)) / 2;
}
else {
if(det == 0) {
ans_ins.num = 1;
ans_ins.ans1 = (-a + sqrt(det)) / 2;
}
else {
ans_ins.num = 0;
}
}
}

int main()
{
float a, b;

printf("a = ");
scanf("%f", &a);

printf("b = ");
scanf("%f", &b);

solve(a, b);

switch(ans_ins.num) {
case 0:
printf("実数解なし\n");
break;
case 1:
printf("重解\n");
printf("解 = %f\n", ans_ins.ans1);
break;
case 2:
printf("相異なる2つの実数解\n");
printf("解1 = %f\n", ans_ins.ans1);
printf("解2 = %f\n", ans_ins.ans2);
break;
}

return 0;
}