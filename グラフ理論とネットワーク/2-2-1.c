//グラフ理論とネットワーク
//BP22104 松本優瑞
//2025/06/19
#include<stdio.h>
#include<stdlib.h>

#define MAX_N 1000

int N; 
int adj[MAX_N][MAX_N];
int visited[MAX_N];

void dfs(int v){
    visited[v] = 1;
    for(int i = 0; i < N; i++){
        if(adj[v][i] && !visited[i]){
            dfs(i);
        }
    }
}

//辺のある頂点がすべて訪問可能か判断
int is_connected(){
    int start = -1;
    for(int i = 0; i < N; i++){
        int deg = 0;
        for(int j = 0; j < N; j++){
            if (adj[i][j]) deg++;
        }
        if (deg > 0){
            start = i;
            break;
        }
    }
    if(start == -1) return 1;

    for(int i = 0; i < N; i++) visited[i] = 0;
    dfs(start);

    for(int i = 0; i < N; i++){
        int deg = 0;
        for(int j = 0; j < N; j++){
            if(adj[i][j]) deg++;
        }
        if(deg > 0 && !visited[i]) return 0;
    }
    return 1;
}

int all_even_degree(){
    for(int i = 0; i < N; i++){
        int deg = 0;
        for(int j = 0; j < N; j++){
            if(adj[i][j]) deg++;
        }
        if(deg % 2 != 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]){

    FILE* fp = fopen(argv[1], "r");
    if (!fp){
        printf("File open error \n");
        return 1;
    }

    fscanf(fp, "%d", &N);

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
          adj[i][j] = 0;
    
    int u, v;

    while (fscanf(fp, "%d %d", &u, &v) == 2)
    {
        adj[u][v] = 1;
        adj[v][u] = 1;
    }
    fclose(fp);

    if(is_connected() && all_even_degree()){
        printf("This is Euler graph\n");
    }else{
        printf("This is not Euler graph\n");
    }

    return 0;
    
}