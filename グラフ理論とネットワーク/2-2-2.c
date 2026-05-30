//グラフ理論とネットワーク
//BP22104 松本優瑞
//2025/06/22

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define MAX 100

int graph[MAX][MAX]; //隣接行列
int degree[MAX]; //各頂点の次数
int V;

void removeEdge(int u, int v){
    graph[u][v]--;
    graph[v][u]--;
}

void addEdge(int u, int v){
    graph[u][v]++;
    graph[v][u]++;
}

void dfsCount(int v, int visited[]){
    visited[v] = 1;
    for (int i = 0; i < V; i++){
        if(graph[v][i] && !visited[i]){
            dfsCount(i, visited);
        }
    }
}

int isValidNextEdge(int u, int v){
    int count = 0;
    for(int i = 0; i < V; i++)
        if(graph[u][i]) count++;
    
    if(count == 1) return 1;

    int visited[MAX] = {0};
    dfsCount(u, visited);
    int countl = 0;
    for(int i = 0; i < V; i++) if (visited[i]) countl++;

    removeEdge(u, v);
    memset(visited, 0, sizeof(visited));
    dfsCount(u, visited);
    int count2 = 0;
    for(int i = 0; i < V; i++) if (visited[i]) count2++;

    addEdge(u, v);

    return(countl <= count2);
}

void printEulerUtil(int u){
    for(int v = 0; v < V; v++){
        if(graph[u][v] && isValidNextEdge(u, v)) {
            printf("%d -> %d\n", u, v);
            removeEdge(u, v);
            printEulerUtil(v);
        }
    }
}

int isEulerian(){
    for(int i = 0; i < V; i++)
      if(degree[i] % 2 != 0)
      return 0;
    
      return 1;
}

int isConnected(){
    int visited[MAX] = {0};
    for(int i = 0; i < V; i++){
        if (degree[i] != 0){
            dfsCount(i, visited);
            break;
        }
    }
    
    for(int i = 0; i < V; i++){
        if(degree[i] != 0 && !visited[i])
         return 0;
    }
    return 1;
}

void loadGraph(const char *filename){
    FILE *fp = fopen(filename, "r");
    if(!fp){
        perror("File open failed");
        exit(1);
    }
    int u, v;
    fscanf(fp, "%d", &V);   
    while (fscanf(fp, "%d %d", &u, &v) != EOF)
    {
        graph[u][v]++;
        graph[v][u]++;
        degree[u]++;
        degree[v]++;
    }
    fclose(fp);

}

int main(int argc, char *argv[]){

    loadGraph(argv[1]);

    if(!isConnected()){
        printf("The graph is not connected, so it does not have an Eulerian circuit.\n");
        return 1;
    }

    if(!isEulerian()){
        printf("The graph is not an Eulerian graph.\n");
        return 1;
    }

    printf("This is Euler graph:\n");
    printEulerUtil(0); // 開始頂点は任意
    return 0;
}