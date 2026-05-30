//グラフ理論とネットワーク
//BP22104 松本優瑞
//2025/7/02
//3-2
#include<stdio.h>
#include<stdlib.h>
#include<float.h> 



int main(int argc, char *argv[]) {
  int i, j;                   // 繰り返し用
  int n1, n2;                 // 一時的に点番号を記憶する変数
  int N;                  // 点の数 
  int **adjacent;
  double w;
  double **dist;
  
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

  //動的メモリを確保
  //演習１で用いたものを使用
  adjacent = (int **)malloc(sizeof(int *) * N);
  dist = (double **)malloc(sizeof(double *) *N);
  for (i = 0; i < N; i++){
    adjacent[i] = (int *)malloc(sizeof(int) * N);
    dist[i] = (double *)malloc(sizeof(double) * N);
  }
   //隣接行列と距離行列を初期化 
    for(i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            adjacent[i][j] = 0;
            dist[i][j] = DBL_MAX;
        }
    }

    //辺情報をを読み込む
  while (fscanf(fp, "%d %d %lf", &n1, &n2, &w) != EOF)
    {
        adjacent[n1][n2] = adjacent[n2][n1] = 1;
        dist[n1][n2] = dist[n2][n1] = w;
        
    }
    
  // ファイルを閉じる
  fclose(fp);

  //Prim法
int *selected = (int *)malloc(sizeof(int) * N);
double *mincost = (double *)malloc(sizeof(double) * N);
int *prev = (int *)malloc(sizeof(int) * N);

for (i = 0; i < N; i++){
    selected[i] = 0;
    mincost[i] = DBL_MAX;
    prev[i] = -1;
}
int start = 0;
mincost[start] = 0;

for (i = 0; i < N; i++){
    double min = DBL_MAX;
    int u = -1;

    for (j = 0; j < N; j++){
        if(!selected[j] && mincost[j] < min){
            min = mincost[j];
            u = j; 
        }
    }

    if (u == -1) break;
    selected[u] = 1;

    for (j = 0; j < N; j++){
        if (adjacent[u][j] && !selected[j] && dist[u][j] < mincost[j]){
            mincost[j] = dist[u][j];
            prev[j] = u;
        }
    }
}

//MSTの辺の出力
printf("Edges in the MST: \n");
for (i = 0; i < N; i++){
    if (prev[i] != -1){
        printf("%d - %d: %1f\n", prev[i], i, dist[prev[i]][i]);
    }
}

//メモリ解放
for(i = 0; i < N; i++){
    free(adjacent[i]);
    free(dist[i]);
}

free(adjacent);
free(dist);
free(selected);
free(mincost);
free(prev);

return 0;
}