import math
import random

def decompose(n):
    q = n - 1
    k = 0
    while q % 2 == 0:
        k += 1
        q //= 2
    return k, q

def miller_rabin_once(n, b):
    k, q = decompose(n)
    x = pow(b, q, n)
    xs = [x]

    if x == 1 or x == n - 1:
        return True, xs

    for j in range(1, k + 1):
        x = pow(x, 2, n)
        xs.append(x)

        if j <= k - 1 and x == n - 1:
            return True, xs

    return False, xs

n = 21
success_count = 0
total = 0

for b in range(2, n - 1):
    result, xs = miller_rabin_once(n, b)
    total += 1

    if result:
        success_count += 1

    print(f"b={b}, 判定={result}, x={xs}")

print("失敗確率 =", success_count, "/", total)
print("失敗確率 =", success_count / total)