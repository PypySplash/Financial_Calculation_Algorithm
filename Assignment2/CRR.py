import numpy as np
# import time
def CRR_binomial_tree_model(S0, K, r, q, sigma, T, n, callorput, AorE):
    dt = T / n
    u = np.exp(sigma * dt ** 0.5)
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    # stock price
    S = np.zeros([n + 1, n + 1])
    # intrinsic value
    iv = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(i + 1):
            S[j, i] = S0 * d ** j * u ** (i - j)
            if callorput == 'call':  # 1 for call
                iv[j, i] = max(S[j, i] - K, 0)
            elif callorput == 'put':  # 0 for put
                iv[j, i] = max(K - S[j, i], 0)

    # present value
    pv = np.zeros([n + 1, n + 1])
    pv[:, n] = iv[:, n]
    for i in np.arange(n - 1, -1, -1):
        for j in np.arange(0, i + 1):
            if AorE == 'E':  # 1 for European
                pv[j, i] = df * (p * pv[j, i + 1] + (1 - p) * pv[j + 1, i + 1])
            elif AorE == 'A':  # 0 for American
                pv[j, i] = max(df * (p * pv[j, i + 1] + (1 - p) * pv[j + 1, i + 1]), iv[j, i])
    #     return pv[:,:]
    return pv[0, 0]


# test = binomial_tree_call_put(S0=100, K=90, r=0.05, q=0.02, sigma=0.5, T=0.5, n=100, callorput='call', AorE='E')
# print('n=100:\n', np.matrix(test.astype(float)))

Ecall = CRR_binomial_tree_model(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=100, callorput='call', AorE='E')
Eput = CRR_binomial_tree_model(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=100, callorput='put', AorE='E')
Acall = CRR_binomial_tree_model(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=100, callorput='call', AorE='A')
Aput = CRR_binomial_tree_model(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=100, callorput='put', AorE='A')

print('Ecall:',Ecall)
print('Eput:',Eput)
print('Acall:',Acall)
print('Aput:',Aput)
