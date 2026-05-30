#include<stdio.h>
#include<stdlib.h>
//参考資料　犯罪統計　凶悪犯の部分
struct police{
	char Year[5];
    int N;
    int K;
	int KK;
    
};

int main(void){
	int i;
    struct police a[5], *p;
    p = a;
    
	for(i=0;i<5;i++){
		printf("何年か入力してください→");
		scanf("%s", (p+i)->Year);
		printf("認知件数:");
		scanf("%d", &(p+i)->N);
		printf("検挙件数：");
		scanf("%d", &(p+i)->K);
		printf("検挙人数:");
		scanf("%d", &(p+i)->KK);

	}
for (i=1; i<5; i++)
{
 printf("%d番目:%s\n", i, (p+i)->Year);
 printf("認知件数:%d\n",i,(p+i)->N);
 printf("検挙件数:%d\n",i,(p+i)->K);
 printf("検挙人数:%d\n",i,(p+i)->KK);

}
return 0;
}