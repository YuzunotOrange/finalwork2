import numpy as np

def second_derivative_test():
    # x(t) = t^3
    def x(t):
        return t**3
    
    def x_double_prime_theory(t):
        return 6 * t
    
    t = 2.0     #評価点
    dt = 0.01   #変化量(Δt)

    #差分近似解の計算
    xn_plus_1 = x(t + dt)
    xn = x(t)
    xn_minus_1 = x(t - dt)

    approximation = (xn_plus_1 - 2 * xn + xn_minus_1) / (dt**2)
    theory = x_double_prime_theory(t)


    print(f"t = {t}, Δt = {dt}")
    print(f"真の解: {theory:.6f}")
    print(f"差分近似値: {approximation:.6f}")
    print(f"誤差: {abs(theory - approximation):.6e}")

if __name__ =="__main__":
    second_derivative_test()
