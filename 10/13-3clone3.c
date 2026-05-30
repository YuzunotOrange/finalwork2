#include<stdio.h>
int main(void){
    FILE *fp;
    char fname[70] = "Hitman.txt";
    int i;
    char name[20];
    char name2[20];
    char age[20];
    char profession[20];
    char family[20];
    char remark[20];
   fp = fopen("Hitman.txt", "r"); // ファイルを開く。失敗するとNULLを返す。
	if(fp == NULL) {
		printf("%s file not open!\n", fname);
		return -1;
	} else {
		printf("%s file opened!\n", fname);
	}
    for(i = 0; i < 70; i++)
    fscanf(fp, "%s %s %s %s %s %s",name[i],name2[i],age[i],profession[i],family[i],remark[i]);
    for(i = 0; i < 70; i++)
    printf("%s\n %s\n %s\n %s\n %s\n %s\n",name[i],name2[i],age[i],profession[i],family[i],remark[i]);
    fclose( fp );
    return 0;
    
}
