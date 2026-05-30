/*QUiz12.c
  2022-12-28
  BP22104
  松本優瑞*/

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
	struct police x[6];
	int i;
	for(i=0;i<5;i++){
		printf("何年か入力してください→");
		scanf("%s", x[i].Year);
		printf("認知件数:→");
		scanf("%d",&x[i].N);
		printf("検挙数：");
		scanf("%d",&x[i].K);
		printf("検挙件数:");
		scanf("%d",&x[i].KK);

	}
for (i=0; i<5; i++)
{
 printf("%d番目:%s\n", i, x[i].Year);
 printf("認知件数：%d\n 検挙数：%d\n 検挙人数:%d\n", x[i].N, x[i].K, x[i].KK);

}
return 0;
}