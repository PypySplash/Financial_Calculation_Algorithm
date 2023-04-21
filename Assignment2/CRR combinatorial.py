import numpy as np

def sum(x):
    summation = 0
    for i in range(1, x + 1):
        summation += np.log(i)
    return summation

def CRR_combinatorial(S0, K, r, q, sigma, T, n, callorput):
    dt = T / n
    u = np.exp(sigma * dt ** 0.5)
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    option_value = 0
    if callorput == 'call':
        for j in range(n + 1):
            combinatorial = sum(n) - sum(n - j) - sum(j) + (n - j) * np.log(p) + j * np.log(1 - p)
            option_value += np.exp(-r * T) * np.exp(combinatorial) * max(S0 * u ** (n - j) * d ** j - K, 0)
    if callorput == 'put':
        for j in range(n + 1):
            combinatorial = sum(n) - sum(n - j) - sum(j) + (n - j) * np.log(p) + j * np.log(1 - p)
            option_value += np.exp(-r * T) * np.exp(combinatorial) * max(K - S0 * u ** (n - j) * d ** j, 0)

    return option_value

call_price = CRR_combinatorial(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=10000, callorput='call')
put_price = CRR_combinatorial(S0=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, n=10000, callorput='put')

print("call price:", call_price)
print("put price:", put_price)
