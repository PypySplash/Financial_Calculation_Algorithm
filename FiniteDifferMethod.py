import numpy as np


def FDM_implicit(S0, K, r, q, sigma, T, Smin, Smax, m, n, callorput, EorA):
    dt = T/n

    def payoff(S):
        if callorput == 'call':
            return max(S-K, 0)
        if callorput == 'put':
            return max(K-S, 0)

    # 建立payoff array
    payoff_array = np.zeros([m-1, n+1])
    # 輸入boundary condition
    for i in range(1, m):  # i不從0是因為 S(0,n) = Smax
        Sij = Smax * i / m
        payoff_array[(m-1)-i][n] = payoff(Sij)
    # print(price_array)

    def a(j):
        return (r-q)/2 * j * dt - 0.5 * sigma**2 * j**2 * dt
    def b(j):
        return 1 + sigma**2 * j**2 * dt + r * dt
    def c(j):
        return -(r-q)/2 * j * dt - 0.5 * sigma**2 * j**2 * dt

    # 生成差分關係矩陣A
    A = np.zeros([m-1, m-1])
    A[0][0] = b(m-1)
    A[0][1] = a(m-1)
    A[m-2][m-3] = c(1)
    A[m-2][m-2] = b(1)
    for i in range(1, m-2):
        A[i][i-1] = c(m-1-i)
        A[i][i] = b(m-1-i)
        A[i][i+1] = a(m-1-i)
    A_inv = np.linalg.inv(A)

    for i in range(n):
        B = payoff_array[:, n-i].T
    # print(B)
        B[0] -= c(m-1) * payoff(Smax)
        B[m-2] -= a(1) * payoff(Smin)
        fi = np.matmul(A_inv, B)  # Ainv * B
        if EorA == 'A':
            for j in range(m-1):
                fi[j] = max(fi[j], payoff(Smax * ((m - 1) - j) / m), 0)
        payoff_array[:, (n-1)-i] = fi
    return payoff_array[(m-2)//2][0]  # (m-2)//2才可調整至中間S0的位置


print(f"Implicit European call: {FDM_implicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=400, n=100, callorput='call', EorA='E')}")
print(f"Implicit European put: {FDM_implicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=400, n=100, callorput='put', EorA='E')}")
print(f"Implicit American call: {FDM_implicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=400, n=100, callorput='call', EorA='A')}")
print(f"Implicit American put: {FDM_implicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=400, n=100, callorput='put', EorA='A')}")


def FDM_explicit(S0, K, r, q, sigma, T, Smin, Smax, m, n, callorput, EorA):
    dt = T / n

    def payoff(S):
        if callorput == 'call':
            return max(S - K, 0)
        if callorput == 'put':
            return max(K - S, 0)

    # 建立payoff array
    payoff_array = np.zeros([m + 1, n + 1])
    # 輸入boundary condition
    for i in range(m+1):
        Sij = Smax * i / m
        payoff_array[m - i][n] = payoff(Sij)
    # print(price_array)

    def a(j):
        return 1/(1+r*dt) * ((-0.5 * (r-q) * j * dt) + 0.5 * sigma**2 * j**2 * dt)
    def b(j):
        return 1/(1+r*dt) * (1 - sigma**2 * j**2 * dt)
    def c(j):
        return 1/(1+r*dt) * ((0.5 * (r-q) * j * dt) + 0.5 * sigma**2 * j**2 * dt)

    # 生成差分關係矩陣A
    A = np.zeros([m - 1, m + 1])
    for i in range(m-1):
        A[i][i] = c(m - 1 - i)
        A[i][i+1] = b(m - 1 - i)
        A[i][i+2] = a(m - 1 - i)

    for i in range(n):
        B = payoff_array[:, n - i].T
        fi = np.matmul(A, B)
        if EorA == 'A':
            for j in range(m-1):
                fi[j] = max(fi[j], payoff(Smax * ((m - 1) - j) / m), 0)
        payoff_array[1:m, n-i-1] = fi
        payoff_array[0, n-i-1] = payoff(Smax)
        payoff_array[m, n-i-1] = payoff(Smin)
    return payoff_array[m // 2][0]


print(f"Explicit European call: {FDM_explicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=100, n=1000, callorput='call', EorA='E')}")
print(f"Explicit European put: {FDM_explicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=100, n=1000, callorput='put', EorA='E')}")
print(f"Explicit American call: {FDM_explicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=100, n=1000, callorput='call', EorA='A')}")
print(f"Explicit American put: {FDM_explicit(S0=50, K=50, r=0.05, q=0.01, sigma=0.4, T=0.5, Smax=100, Smin=0, m=100, n=1000, callorput='put', EorA='A')}")
