#include<stdio.h>
int recursive_combinatin(int, int);
int main(void)
{
    int n,r;
    do
    {
        printf("0以上の整数値を入力してください→");
        scanf("%d %d", &n, &r);
    } while (n < r);
    printf("計算結果：%d\n",  recursive_combinatin( n, r));
    return 0;
}
int  recursive_combinatin(int n, int r){
    if(r == 0 || r == n){
    return 1;    
    }
     else if ( r == 1){
            
            return n;
        }
        return recursive_combinatin(n - 1,r) + recursive_combinatin(n - 1,r - 1);

        }
    