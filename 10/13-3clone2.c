#include<stdio.h>
#include<string.h>
int main(void) {
     FILE *fp;
    char fname[80] = "Hitman.txt";
    char searchname[100];
    char prev[1000];
    char str[1000] = { '\0'};
   fp = fopen("Hitman.txt", "r"); // ファイルを開く。失敗するとNULLを返す。
	if(fp == NULL) {
		printf("%s file not open!\n", fname);
		return -1;
	} else {
		printf("%s file opened!\n", fname);
	}
    printf("名前を検索:");
    scanf("%s",searchname);
    while (1)
    {
        strcpy(prev, str);                    
        if(strstr(str, searchname) != NULL){
        printf("%s", prev);
        printf("%s", str);
        fgets(str, sizeof(str), fp); printf("%s",str);
        fgets(str, sizeof(str), fp); printf("%s",str);
        fgets(str, sizeof(str), fp); printf("%s",str);
        fgets(str, sizeof(str), fp); printf("%s",str);
        fgets(str, sizeof(str), fp); printf("%s",str);
        
    }
    }
    fclose(fp);
    return 0;
}
