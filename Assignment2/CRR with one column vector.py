import numpy as np
# import time
def CRR_with_1_dim(S0, K, r, q, sigma, T, n, callorput, AorE):
    dt = T / n
    u = np.exp(sigma * dt ** 0.5)
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    sp_list = np.zeros(n + 1)
    iv_list = np.zeros(n + 1)
    for i in range(n + 1):
        sp_list[i] = S0 * u ** i * d ** (n - i)
        if callorput == 'call':
            iv_list[i] = max(sp_list[i] - K, 0)
        elif callorput == 'put':
            iv_list[i] = max(K - sp_list[i], 0)
    #     print(sp_list)
    #     print(iv_list)
    pv_list = []
    pv_list[:] = iv_list[:]

    for j in np.arange(n - 1, -1, -1):
        for i in np.arange(0, j + 1):
            if AorE == 'E':  # E for European
                pv_list[i] = df * (p * pv_list[i + 1] + (1 - p) * pv_list[i])
            elif AorE == 'A':  # A for American
                sp_list[i] = S0 * u ** i * d ** (j - i)
                if callorput == 'call':
                    iv_list[i] = max(sp_list[i] - K, 0)
                elif callorput == 'put':
                    iv_list[i] = max(K - sp_list[i], 0)
                pv_list[i] = max(df * (p * pv_list[i + 1] + (1 - p) * pv_list[i]), iv_list[i])
    price = pv_list[0]
    return price

Ecall = CRR_with_1_dim(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=500, callorput='call', AorE='E')
Eput = CRR_with_1_dim(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=500, callorput='put', AorE='E')
Acall = CRR_with_1_dim(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=500, callorput='call', AorE='A')
Aput = CRR_with_1_dim(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=500, callorput='put', AorE='A')

print('Ecall:', Ecall)
print('Eput:', Eput)
print('Acall:', Acall)
print('Aput:', Aput)