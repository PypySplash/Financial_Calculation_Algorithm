import numpy as np

def Cheuk_Vorst_lookback_put(St, r, q, sigma, t, T, n, Smaxt, AorE):
    dt = (T - t) / n
    u = np.exp(sigma * dt ** 0.5)
    d = 1 / u
    mu = np.exp((r - q) * dt)
    p = (mu * u - 1) / (mu * (u - d))
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt) * mu

    # 創一個u的list
    u_list = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(i + 1):
            u_list[j,i] = u ** (i-j)
    # print(u_list)
    payoff_list = np.zeros([n+1, n+1])
    for j in range(n+1):
        payoff_list[j,n] = max(u_list[j,n] - 1, 0)
    # print(payoff_list)
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            if AorE == 'E':
                if i == j:
                    payoff_list[j,i] = df * ((1-p) * payoff_list[j][i+1] + p * payoff_list[j+1][i+1])
                if i != j:
                    payoff_list[j,i] = df * ((1-p) * payoff_list[j][i+1] + p * payoff_list[j+2][i+1])
            if AorE == 'A':
                if i == j:
                    payoff_list[j,i] = max((1-p) * payoff_list[j][i+1] + p * payoff_list[j+1][i+1], u_list[j,i] - 1)
                if i != j:
                    payoff_list[j,i] = max((1-p) * payoff_list[j][i+1] + p * payoff_list[j+2][i+1], u_list[j,i] - 1)
    # print(payoff_list)
    return(St * payoff_list[0,0])

Eput1000 = Cheuk_Vorst_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=1000, Smaxt=50, AorE='E')
print(f"European puts n=1000: {Eput1000}")
Aput1000 = Cheuk_Vorst_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=1000, Smaxt=50, AorE='A')
print(f"American puts n=1000: {Aput1000}")
