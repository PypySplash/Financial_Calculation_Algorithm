import numpy as np

def cholesky_decomposition(K, T, r, ns, nr, n, S, q, sigma, rho):
    # miu = ln(S0*exp((r-q-sigma**2/2)*T))
    miu = []
    for i in range(n):
        miui = np.log(S) + (r - q - sigma ** 2 / 2) * T
        miu.append(miui)
    #     print(miu)
    payoff_mean = []
    for repeat in range(nr):
        # Covariance Matrix
        C = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    C[i][j] = sigma ** 2 * T
                if i < j:
                    C[i][j] = rho * sigma * sigma * T
                else:
                    C[i][j] = C[j][i]
        #         print(C)
        # Cholesky decomposition
        U = np.zeros([n, n])  # U: upper上三角矩陣
        sum_akikj = 0
        for i in range(n):
            for j in range(i, n):
                sum_akikj = sum(U[k][i] * U[k][j] for k in range(i))
                if i == j:
                    U[i][j] = (C[i][i] - sum_akikj) ** 0.5
                else:
                    U[i][j] = 1.0 / U[i][i] * (C[i][j] - sum_akikj)
        #         print(U)
        # 抽N(0,1)放入Z矩陣
        z = np.zeros([ns, n])
        for i in range(ns):
            for j in range(n):
                z[i][j] = np.random.normal()
        rj = z.dot(U)  # z*U = r
        # r取exp即為ST
        S_list = np.zeros([ns, n])
        for i in range(ns):
            for j in range(n):
                rj[i][j] += miu[j]
                S_list[i][j] = np.exp(rj[i][j])
        #         print(S_list)
        # rainbow option
        payoff_list = []
        for i in range(ns):
            max_payoff = 0
            for j in range(n):
                payoff = max(S_list[i][j] - K, 0)
                if max(S_list[i][j] - K, 0) > max_payoff:
                    max_payoff = max(S_list[i][j] - K, 0)
            discount_payoff = np.exp(-r * T) * max_payoff
            payoff_list.append(discount_payoff)
        payoff_mean.append(np.mean(payoff_list))
    payoff = np.mean(payoff_mean)
    payoff_se = np.std(payoff_mean)
    payoff_down = payoff - 2 * payoff_se
    payoff_up = payoff + 2 * payoff_se
    print(f"Rainbow maximum call:{payoff}")
    print(f"95%C.I:[{payoff_down}, {payoff_up}]")

cholesky_decomposition(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                       n = 2, S = 95, q = 0.05, sigma = 0.5, rho = 1)
cholesky_decomposition(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                       n = 2, S = 95, q = 0.05, sigma = 0.5, rho = -1)
cholesky_decomposition(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                       n = 5, S = 95, q = 0.05, sigma = 0.5, rho = 0.5)
