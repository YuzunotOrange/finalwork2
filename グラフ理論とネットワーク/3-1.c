//グラフ理論とネットワーク
//BP22104 松本優瑞
//2025/06/23

#include<stdio.h>
#include<stdlib.h>
#include<float.h> 

#define MAX_WEIGHT 1.0e30

//最小距離のノードを取得
int minDistance(double *dist, int *visited, int N){
    double min = MAX_WEIGHT;
    int min_index = -1;
    for (int v = 0; v < N; v++){
        if(!visited[v] && dist[v] <= min){
            min = dist[v];
            min_index = v;
        }
    }

    return min_index;
}

//ダイクストラのアルゴリズム
double dijkstra(int n, double **dist, int start, int end){
    double *distance; //L(v) 距離ラベル
    int *parent; //parent(v)　親ラベル
    int *visited; //U　最短経路確定
    int i, x, y;
    double d;

    distance = (double *)malloc(sizeof(double) * n);
    parent  = (int *)malloc(sizeof(int) * n);
    visited = (int *)malloc(sizeof(int) * n);

    //初期化
    for(int v = 0; v < n; v++){
        distance[v] = MAX_WEIGHT;
        parent[v] = -1;
        visited[v] = 0;
    }

    //(2) 始点
    distance[start] = 0;

    //(3)
    while ( 1 ){

        //V-Uから距離ラベル最小の点を選択
        x = -1;
        d = MAX_WEIGHT;
        for(i = 0; i < n; i++){
            if (visited[i] == 0 && distance[i] < d){
                x = i;
                d = distance[i];
            }
        }

        //x = t なら終了
        if(x == -1 || x == end)break;
        
        //V-Uに含まれ、かつxに隣接しているすべての点iに対して
        for(y = 0; y < n; y++){
            if(dist[x][y] != MAX_WEIGHT && !visited[y]){
                double alt = distance[x] + dist[x][y];
                if (alt < distance[y]){
                    distance[y] = alt;
                    parent[y] = x;
                }
            }

        }
        visited[x] = 1;
    }

    if(distance[end] == MAX_WEIGHT){
        printf("No path from %d to %d\n", start, end);
    }else{
        printf("Shortest distance from %d to %d: %.6f\n", start, end, distance[end]);
        printf("Patj: ");
        int path[100], k = 0;
        for(int v = end; v != -1; v = parent[v]) path[k++] = v;
        for(int i = k - 1; i >= 0; i--){
            printf("%d", path[i]);
            if(i != 0) printf(" -> ");
        }
        printf("\n");
    }

    free(distance);
    free(visited);
    free(parent);
}


int main(int argc, char *argv[]) {
  int i, j;                   // 繰り返し用
  int n1, n2;                 // 一時的に点番号を記憶する変数
  int N, M = 0;                  // 点の数 辺の数
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
  adjacent = (int **)malloc(sizeof(int *) * N);
  dist = (double **)malloc(sizeof(double *) *N);
  for (i = 0; i < N; i++){
    adjacent[i] = (int *)malloc(sizeof(int) * N);
    dist[i] = (double *)malloc(sizeof(double) * N);
    for(j = 0; j < N; j++){
        adjacent[i][j] = 0;
        if(i == j){
            dist[i][j] = 0;
        } else {
            dist[i][j] = MAX_WEIGHT;
        }
    }
  }
  while (fscanf(fp, "%d %d %lf", &n1, &n2, &w) != EOF)
    {
        adjacent[n1][n2] = adjacent[n2][n1] = 1;
        dist[n1][n2] = dist[n2][n1] = w;
        M++;
    }
    
  // ファイルを閉じる
  fclose(fp);
  
  printf("The number of vertics: %d\n", N);
  printf("The number of edges: %d\n", M);

  printf("Adjacent matrix:\n");
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
        printf("%d ", adjacent[i][j]);
    }
    printf("\n");
  }

  //始点と終点を取得
  int start, end;
  printf("Enter start and end node(e.g, 0 9): ");
  scanf("%d %d", &start, &end);

  dijkstra(N,dist, start, end);

  for(i = 0; i< N; i++){
    free(adjacent[i]);
    free(dist[i]);
  }
  
  free(adjacent);
  free(dist);

  return 0;
}


