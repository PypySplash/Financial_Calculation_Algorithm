import numpy as np

def cholesky_decomposition_by_ava_mmm(K, T, r, ns, nr, n, S, q, sigma, rho):
    miu = []
    for i in range(n):
        miui = np.log(S) + (r - q - sigma ** 2 / 2) * T
        miu.append(miui)
    #     print(miu)
    price_mean = []
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
        # 抽N(0, 1)放入Z矩陣
        # antithetic variate approach(反向變異法)
        z = np.zeros([ns, n])
        for i in range(ns):
            for j in range(n):
                if i <= ns / 2:
                    z[i][j] = np.random.normal()
                else:
                    z[i][j] = -z[i - 5000][j]
        # moment matching method
        zj_std_list = []
        for j in range(n):
            zj_list = []
            for i in range(ns):
                zj_list.append(z[i][j])
            zj_std = np.std(zj_list)
            zj_std_list.append(zj_std)

        y = np.zeros([ns, n])
        for i in range(ns):
            for j in range(n):
                y[i][j] = z[i][j] / zj_std_list[j]
        rj = y.dot(U)

        S_list = np.zeros([ns, n])
        for i in range(ns):
            for j in range(n):
                rj[i][j] += miu[j]
                S_list[i][j] = np.exp(rj[i][j])
        #         print(S_list)

        payoff_list = []
        for i in range(ns):
            max_payoff = 0
            for j in range(n):
                payoff = max(S_list[i][j] - K, 0)
                if max(S_list[i][j] - K, 0) > max_payoff:
                    max_payoff = max(S_list[i][j] - K, 0)
            discount_payoff = np.exp(-r * T) * max_payoff
            payoff_list.append(discount_payoff)
        price_mean.append(np.mean(payoff_list))
    price = np.mean(price_mean)
    price_se = np.std(price_mean)
    price_down = price - 2 * price_se
    price_up = price + 2 * price_se
    print(f"Rainbow maximum call:{price}")
    print(f"95%C.I:[{price_down}, {price_up}]")

cholesky_decomposition_by_ava_mmm(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                                  n = 2, S = 95, q = 0.05, sigma = 0.5, rho = 1)
cholesky_decomposition_by_ava_mmm(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                                  n = 2, S = 95, q = 0.05, sigma = 0.5, rho = -1)
cholesky_decomposition_by_ava_mmm(K = 100, T = 0.5, r = 0.1, ns = 10000, nr = 20, \
                                  n = 5, S = 95, q = 0.05, sigma = 0.5, rho = 0.5)
