//グラフ理論とネットワーク　課題1-3-2
//BP22104 松本優瑞
//2025_05_31


#include<stdio.h>
#include<stdlib.h>

#define N 7 
#define TRIALS 10000

int main(){
    int adj[N][N] = {0};
    int from, to;

    //ファイルを読み込む
    FILE *fp = fopen("graph1.txt", "r");
    
    int node_count;
    fscanf(fp, "%d", &node_count);

    while(fscanf(fp, "%d %d", &from, &to) != EOF){
        adj[from][to] = 1;
        adj[to][from] = 1;
    }

    fclose(fp);

    long long total_hops = 0;

    //試行開始
    for (int trial = 0; trial < TRIALS; trial++){
        int pos = 0;
        int hop_count = 0;
        
    

    //乱数のたね 課題2ではここを変化させる
    //srand48(12345); Linux/Unix形で利用
    srand(20250531 + trial);


    while(pos != 6) {
        int neighbors[N];
        int count = 0;

        for (int i = 0; i < N; i++){
            if (adj[pos][i] == 1){
                neighbors[count++] = i;
            }
        }

        //int r = (int)(drand48() * count);
        int r = (int)((rand() / (RAND_MAX + 1.0)) * count);
        int next = neighbors[r];

        pos = next;
        hop_count++;
    }
    
    total_hops += hop_count;
}
double average = (double)total_hops / TRIALS;
printf("Average number of hops: %.5f\n", average);

    return 0;
}