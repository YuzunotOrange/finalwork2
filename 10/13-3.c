/*13-3.c
2023-1-10
BP22104
松本優瑞*/
#include <stdio.h>
#include <string.h>

int main() {
  FILE* fp;
  fp = fopen("Hitman.txt", "r");
  char prev[1000];
  char str[1000] = { '\0' };
  char searchname[100];
   fp = fopen("Hitman.txt", "r"); // ファイルを開く。失敗するとNULLを返す。
	if(fp == NULL) {
		printf("%s データをファイルから読み込めませんでした!\n", fp);
		return -1;
	} else {
		printf("%s データをファイルから読み込みました!\n", fp);
	}
  printf("Enter search nane: ");
  scanf("%s", searchname);
  while ( 1 ) {
    strcpy(prev, str);
    if ( fgets(str, sizeof(str), fp) == NULL ) break;
    // 検索文字列を見つけたら
    if (strstr(str, searchname) != NULL) {
      // 直前の行、その行、さらに後続する4行をプリント
      printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
      fgets(str, sizeof(str), fp); printf("%s", str);
    }
  }
  fclose(fp);
  return 0;
}