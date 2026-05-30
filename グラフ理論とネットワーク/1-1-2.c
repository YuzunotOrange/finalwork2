
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int i, j;                   // 繰り返し用
  int n1, n2;                 // 一時的に点番号を記憶する変数
  int N;                      // 点の数
  int M = 0;                  // 辺の数
  int **adjacent;             // 隣接行列を格納する2次元ポインタ
  
  FILE *fp;

  // 実行時引数のチェック
  if (argc != 2) {
    fprintf(stderr, "Usage: %s graph_file\n", argv[0]);
    exit(1);
  }

  // ファイルを開く
  if((fp=fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Cannot open file %s\n", argv[1]);
    exit(1);
  }
  
  // 点の数を読み込む
  fscanf(fp, "%d", &N);

  // 配列を動的に確保する
  adjacent = (int **)malloc(sizeof(int *) * N);
  for (i=0; i<N; i++) {
    adjacent[i] = (int *)malloc(sizeof(int) * N);
  }

  // 隣接行列を0でクリアする
  for (i=0; i<N; i++) {
    for (j=i; j<N; j++) {
      adjacent[i][j] = adjacent[j][i] = 0;
    }
  }

  
  // ファイルの最後まで辺情報を読み込む
  while(fscanf(fp, "%d %d", &n1, &n2) != EOF) {
    // 辺のある個所の隣接行列を1に変更
    adjacent[n1][n2] = 1;
    adjacent[n2][n1] = 1;
    // 辺の数を1増やす
    M++;
  }

 
  // 点の数を表示
  printf("The number of vertices: %2d\n", N);

  // 辺の数を表示
  printf("The number of edges:    %2d\n", M);

  // 隣接行列を表示
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      printf( "%d ", adjacent[i][j]);
    }
    printf("\n");
  }

  // 配列を開放する
  for (i=0; i<N; i++) {
    free(adjacent[i]);
  }
  free(adjacent);
  
  // ファイルを閉じる
  fclose(fp);
  
  return 0;
}