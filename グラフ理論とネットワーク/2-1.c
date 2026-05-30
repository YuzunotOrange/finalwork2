//グラフ理論とネットワーク
//bp22104 松本優瑞
//2025/06/18
#include<stdio.h>
#include<stdlib.h>


#define N 8

int** adjacent;


int connect_check(int size, int **mattix){
    int visited[N] = {0};
    int queue[N], front, rear;
    int components = 0;

    for (int i = 0; i < N; i++){
        if (visited[i]) continue;
    

    //有効なノードかを確認
    int vaild = 0;
    for(int j = 0; j < N; j++){
        if(mattix[i][j]){
            vaild = 1;
            break;
        }
    }
    if (!vaild) continue;

    components++;
    visited[i] = 1;
    front = rear = 0;
    queue[rear++] = i;
    

    while(front < rear){
        int node = queue[front++];
        for(int j = 0; j < N; j++){
            if(mattix[node][j] && !visited[j]){
                visited[j] = 1;
                queue[rear++] = j;
            }
            }
        }
    }
    return components;
}




int min_degree(){
    int min = N;
    for(int i = 0; i < N; i++){
        int deg = 0;
        for(int j = 0; j < N; j++){
            if(adjacent[i][j]) deg++;
        }
        if(deg < min) min = deg;
    }
    return min;
}


void print_set(int val){
    for(int i = 0; i < N; i++){
        if((val >> i) & 1)
        printf("%d ", i);
    }
}

int is_separating_set(int val, int original_components){
    int **tmp_adj = (int **)malloc(sizeof(int *) * N);
    for(int i = 0; i < N; i++){
        tmp_adj[i] = (int *)malloc(sizeof(int) * N);
        for(int j = 0; j < N; j++){
            if(((val >> i) & 1) || ((val >> j) & 1)){
                tmp_adj[i][j] = 0;
            }else{
                tmp_adj[i][j] = adjacent[i][j];
            }
        }
    }
    int modified_components = connect_check(N, tmp_adj);

    for(int i = 0; i < N; i++) free(tmp_adj[i]);
    free(tmp_adj);

    return (original_components == 1 && modified_components > 1);
}

int main(int argc, char *argv[]){
    //メモリを確保、隣接行列の初期化
    adjacent = (int **)malloc(sizeof(int *) * N);
    for ( int i = 0; i < N; i++){
        adjacent[i] = (int *)malloc(sizeof(int) * N);
    }

    FILE* fp = fopen(argv[1], "r");
    if (!fp){
        printf("ファイルが開けません。 \n");
        return 1;
    }
    int dummy;
    fscanf(fp, "%d", &dummy);

    for (int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
     adjacent[i][j] = 0;
    
    //辺リストから隣接行列を作成
    int a, b;
    while(fscanf(fp, "%d %d", &a, &b) == 2){
        adjacent[a][b] = 1;
        adjacent[b][a] = 1;
    }
    fclose(fp);

    int delta = min_degree();

    int original_components = connect_check(N, adjacent);

    for (int x = 1; x <= delta; x++){
        printf("***** x = %d *****\n", x);
        for(int val = 1; val < (1 << N); val++){
            int count = 0;
            for(int i = 0; i < N; i++)
               if((val >> i) & 1) count++;
            if(count != x) continue;

            if(is_separating_set(val, original_components)){
                print_set(val);
                printf("Separated!!\n");
                printf("k(G) = %d\n", x);
                printf("Separating set = ( ");
                print_set(val);
                printf(")\n");

                //メモリを解放して終了
                for(int i = 0; i < N; i++) free(adjacent[i]);
                free(adjacent);
                return 0;
            } else{
                print_set(val);
                printf("Connected\n");
            }
        }
    }
    printf("k(G) = %d\n", delta);
    printf("x = δ、δが点連結度です\n");

    for(int i = 0; i < N; i++) free(adjacent[i]);
    free(adjacent);
    return 0;
}
