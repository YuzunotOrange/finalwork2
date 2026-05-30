/*
グラフ理論とネットワーク
演習課題　第４回
BP22104 松本　優瑞
2025/07/14
*/
#include <stdio.h>
#include <stdlib.h>

#define CAPACITY_MAX   2147483647
#define M_MAX                 100
#define N_MAX                   8


/* 連結性を確認する関数（深さ優先探索のため）
   返り値: なし
   隣接行列だけでは計算できないので，辺容量とフローも引数に追加する */
void visit( int N, int **adjacent, int **capacity, int **flow, int v,
            int *yet, int *parent ) {
  int w;

  yet[v] =  0;

  for ( w=0; w<N; w++ ) {
    if ( adjacent[v][w] == 1 && yet[w] == 1 ) {
      /* 関数作成のポイント
         フロー増加が可能な辺だけを辿るように変更
       */

      if(capacity[v][w] - flow[v][w] > 0){
      parent[w] = v;
      visit(N, adjacent, capacity, flow, w, yet, parent);
      }
    }
  }
}


/* s→tフロー増加道を探索する関数
   返り値: 見つかった場合は1，見つからなければ0を返す */
int find_flow_arg_path( int N, int **adjacent, int **capacity, int **flow,
			int *path, int s, int t ) {
  int i, n;
  int counter = 0;
  int *YetToVisit;
  int *parent;
  int *queue;
  int front, rear;
  
  YetToVisit = ( int * )malloc( sizeof( int ) * N );
  parent = ( int * )malloc( sizeof( int ) * N );
  queue = (int *)malloc( sizeof(int ) * N);

  for(i = 0; i < N; i++){
    YetToVisit[i] = 1;
    parent[i] = -1;
  }

  front = rear = 0;
  queue[rear++] = s;
  YetToVisit[s] = 0;

  while(front < rear){
    int v = queue[front++];
    for(i = 0; i < N; i++){
        if (adjacent[v][i] == 1 && YetToVisit[i] == 1){
            if(capacity[v][i] - flow[v][i] > 0){
                parent[i] = v;
                YetToVisit[i] = 0;
                queue[rear++] = i;
            }
        }
    }
  }

  if(parent[t] == -1){
    free(YetToVisit);
    free(parent);
    free(queue);
    return 0;
  }

  n = 0;
  i = t;
  while(i != -1){
    path[counter++] = i;
    i = parent[i];
  }

  for(i = 0; i < counter / 2; i++){
    int tmp = path[i];
    path[i] = path[counter - i -1];
    path[counter - i -1] = tmp;
  }

  free(YetToVisit);
  free(parent);
  free(queue);
  return 1;

}


/* フロー増加量を算出する関数
   返り値: フロー増加量 */
int calc_increasable_flow( int N, int **capacity, int **flow,
			   int *path, int t ) {
  /*
    関数作成のポイント
    この関数はそれほど難しくないであろう．
    path[]に格納されたフロー増加道を辿り，フローの可能増加量を算出する．
    なお，path[]の終わりには終端点「t」が入っていることに注意．
  */

  int i = 0;
  int e_min = CAPACITY_MAX;

  while(path[i] != t){
    int u = path[i];
    int v = path[i + 1];
    if(capacity[u][v] - flow[u][v] < e_min)
     e_min = capacity[u][v] -flow[u][v];
    i++;
  }

  return e_min;

}

/* フローを増加させる関数
   返り値: なし */
void increase_flow( int N, int **capacity, int **flow,
		    int *path, int t, int e_min ) {
  /*
    関数作成のポイント
    この関数はそれほど難しくないであろう．
    e_minで指定された分だけ，フロー増加道上のフローを増加させる．
  */

  int i = 0;

  while(path[i] != t){
    int u = path[i];
    int v = path[i + 1];
    flow[u][v] += e_min;
    flow[v][u] -= e_min;
    i++;
  }
}


/* フロー増加道を表示する関数
   返り値: なし */
void print_path( int N, int *path, int t ) {
  int n = 0;

  while ( n < N && path[n] != t )
    fprintf( stdout, "%d ", path[n++] );
  fprintf( stdout, "%d ", path[n] );
  fprintf( stdout, "\n" );
}
  


int main( int argc, char *argv[] ) {
  int i, j, N, M = 0, a, b, c;
  int s, t;        /* 始点と終点 */
  int e_min;       /* フロー増加量を記憶する変数 */
  int **adjacent;  /* 隣接行列 */
  int **capacity;  /* 辺容量 */
  int **flow;      /* 辺のフロー */
  int *path;       /* フロー増加道記憶配列 */
  int F = 0;       /* s→tフロー */
  FILE *fp;

  if ( argc != 4 ) {
    fprintf( stderr, "Usage: %s data s_node t_node\n", argv[0] );
    exit( 1 );
  }

  if ( ( fp = fopen( argv[1], "r" ) ) == NULL ) {
    fprintf( stderr, "File open error.\n" );
    exit( 1 );
  }

  s = atoi( argv[2] );
  t = atoi( argv[3] );


  /* 点の数を読み込む */
  if( fscanf(fp, "%d", &N) != 1){
    fprintf(stderr, "File format error \n");
    exit(1);
  }

  /* 指定した始点と終点が存在しない場合はエラー */
  if ( s < 0 || s >= N || t < 0 || t >= N ) {
    fprintf( stderr, "Node should satisfy 0 <= s, t < %d\n", N );
    exit( 1 );
  }
  
  /* 配列を malloc() により動的に確保する */
  adjacent = (int **)malloc(sizeof(int *) * N);
  capacity = (int **)malloc(sizeof(int *) * N);
  flow = (int**)malloc(sizeof(int *) *N);
  for(i = 0; i < N; i++){
    adjacent[i] = (int *)malloc(sizeof(int) * N);
    capacity[i] = (int *)malloc(sizeof(int) * N);
    flow[i] = (int *)malloc(sizeof(int) * N);
    for(j = 0; j < N; j++){
      adjacent[i][j] = 0;
      capacity[i][j] = 0;
      flow[i][j] = 0;
    }
  }
  path = (int *)malloc(sizeof(int) * N);
  
  /* ファイルからデータを読み込む */
  while(fscanf(fp, "%d %d %d", &a, &b, &c) == 3){
    if(a < 0 || a >= N || b < 0 || b >= N){
      fprintf(stderr, "Edge node is out of range: %d %d\n", a, b);
      continue;
    }
    adjacent[a][b] = 1;
    adjacent[b][a] = 1;
    capacity[a][b] = c;
  }

  // s→tフロー増加道が見つからなくなるまで，フロー増加を繰り返す */
  while ( find_flow_arg_path( N, adjacent, capacity, flow, path, s, t ) ) {

    /* calc_increasable_flow()を使ってフロー増加量を算出し，
       e_minに代入する */
    e_min = calc_increasable_flow(N, capacity, flow, path, t);

    /* increase_flow()を使ってフロー増加道に沿ってフローを増加させる */
    increase_flow(N, capacity, flow, path, t, e_min);

    /* s→tフローを増加させる */
    F += e_min;

    /* フロー増加道の確認と増加フローの表示（確認用） */
    print_path( N, path, t );
    fprintf( stdout, "e_min = %d\n", e_min );
  }

  /* 最終表示 */
  fprintf( stdout, "Max Flow (%d, %d) is %d\n", s, t, F );


  /* 配列の開放 */
  for(i = 0; i < N; i++){
    free(adjacent[i]);
    free(capacity[i]);
    free(flow[i]);
  }
  free(adjacent);
  free(capacity);
  free(flow);
  free(path);

  return 0;
}