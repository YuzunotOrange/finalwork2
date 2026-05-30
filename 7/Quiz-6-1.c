#include<stdio.h>
#include<math.h>
#define M k
#define N n
#define O m
int main(void){
    int i,j,h;
    int M,N,O;
    /*k,m,nの値を入力してもらう*/
    printf("k,m,nの値を入力してください：",k,m,n);
    scanf("%d %d %d",&k,&m,&n);
    if(k >= 2){
        if(m >= 2){
            if(n >= 2){
                if(k != m){
                    if(m != n){
    M = k, N = n, O = m;  
int A[M][O],B[O][N],C[M][N];
/*ここでAの行列を決める*/
for(i=0; i<M; i++){
for(j=0; j<O; j++){    
    printf("Aの%d行%d列のデータ：",i + 1, j + 1);
    scanf("%f",&A[i][j]);
};
};
/*ここでBの行列を決める*/
for(i=0; i<O; i++){
for(j=0; j<N; j++){ 
    printf("Bの%d行%d列のデータ：",i + 1, j + 1);
    scanf("%f",&B[i][j]);
};
};
/*行列式の積を計算する*/
int c[M][N];
int vecA[O];
int vecB[O];
int inner_product;
for (i = 0; i < M; i++){
for (j = 0; j < N; j++){
for(h = 0; h < O; h++){
vecA[h] = A[i][h];
vecB[h] = B[h][j];
}
inner_product = 0;
for (h = 0; h < O; h++){
    inner_product += vecA[h] * vecB[h];
}
C[i][j] = inner_product;
}    
}
/*結果を出力*/
for(i = 0; i < M; i++){
    for(j = 0; j < N; j++){ 
        printf("Answer[%d][%d]=%d\n",i ,j ,C[i][j]);
    };
    printf("\n");
};
                    }
                    }
                    }
                    }
                    }
    else{
        printf("error\n");
        
    }
    
return 0;
}