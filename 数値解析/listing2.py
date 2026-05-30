import numpy as np
def function(x):
    # def:関数定義
    # xを引数にする関数functionを定義
    return(np.sqrt(x+1)-np.sqrt(x))

#main
a=1.0e15
#1e15=10^15の指数表現
print(f'{a:.15g}, {function(a):.15f}') #出力を16桁で示す。
