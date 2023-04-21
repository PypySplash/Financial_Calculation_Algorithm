import numpy as np


def lookback_put(St, r, q, sigma, t, T, n, Smaxt, AorE):
    dt = (T - t) / n
    u = np.exp(sigma * dt**0.5)
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    # stock price
    S = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(i + 1):
            S[j, i] = St * d**j * u**(i - j)
    # 將股價為50的地方對齊，才不會運算時出現50.00000000001這種浮點數的誤差
    if (n+1) % 2 == 0:
        S[int((n-1)/2), n-1] = S[0,0]
    if (n+1) % 2 != 0:
        S[int(n/2), n] = S[0,0]
    for i in range(n-2,-1,-1):
        for j in range(i+1):
            S[j][i] = S[j+1][i+2]
#     print(S[:,:])
    # 建立一個3層list來裝每個node的可能Smax
    Smax_list = [ [ [] for i in range(n+1) ] for j in range(n+1) ]
    # 原點為max(St, Smaxt)
    Smax_list[0][0].append(max(St, Smaxt))
    for i in range(n+1):
        for j in range(n+1):
            if i == 0 & j == 0:  # 原點股價
                continue
            if j == i:  # 對角線列Smax = Smax_list[0][0]
                Smax_list[j][i].append(Smax_list[0][0][0])
            if j < i:
                if j == 0:  # 第一列Smax = Smaxt
                    Smax_list[j][i].append(max(S[j,i], Smaxt))
                elif j != 0:
                    for k in range(len(Smax_list[j][i-1])):  # forward-tracking method
                        if Smax_list[j][i-1][k] > S[j][i] and Smax_list[j][i-1][k] not in Smax_list[j][i]:
                            Smax_list[j][i].append(max(Smax_list[j][i-1][k], Smaxt))
                            Smax_list[j][i].sort()
                            Smax_list[j][i].reverse()
                        elif Smax_list[j][i-1][k] <= S[j][i] and S[j][i] not in Smax_list[j][i]:
                            Smax_list[j][i].append(max(S[j][i], Smaxt))
                            Smax_list[j][i].sort()
                            Smax_list[j][i].reverse()
                    for k in range(len(Smax_list[j-1][i-1])):
                        if Smax_list[j-1][i-1][k] > S[j][i] and Smax_list[j-1][i-1][k] not in Smax_list[j][i]:
                            Smax_list[j][i].append(max(Smax_list[j-1][i-1][k], Smaxt))
                            Smax_list[j][i].sort()
                            Smax_list[j][i].reverse()
                        elif Smax_list[j-1][i-1][k] <= S[j][i] and S[j][i] not in Smax_list[j-1][i-1]:
                            Smax_list[j][i].append(max(S[j][i], Smaxt))
                            Smax_list[j][i].sort()
                            Smax_list[j][i].reverse()
    # print(Smax_list)
    # 建立一個payoff的三層list
    payoff_list = [ [ [] for i in range(n+1) ] for j in range(n+1) ]
    for j in range(n+1):
        for k in range(len(Smax_list[j][n])):
            if Smax_list[j][n][k] >= S[j][n]:
                payoff = Smax_list[j][n][k] - S[j][n]
                payoff_list[j][n].append(payoff)
            else:
                payoff = S[j][n] - Smax_list[j][n][k]
                payoff_list[j][n].append(payoff)
    # print(payoff_list)
    payoffu = -1
    payoffd = -1
    for i in range(n-1, -1, -1):
        # print(f"跑到第{i}圈")
        for j in range(i+1):
            a = 0
            b = 0
            for k in range(len(Smax_list[j][i])):  # backward induction
                index = 0
                if AorE == 'E':
                    for k1 in range(a, len(Smax_list[j][i+1])):
                        if Smax_list[j][i+1][k1] == Smax_list[j][i][k]:
                            payoffu = payoff_list[j][i+1][k1]
                            a = k1
                            index = 1
                    if index == 0:
                        payoffu = payoff_list[j][i + 1][-1]
                    for k2 in range(b, len(Smax_list[j+1][i+1])):
                        if Smax_list[j+1][i+1][k2] == Smax_list[j][i][k]:
                            payoffd = payoff_list[j+1][i+1][k2]
                            b = k2
                    payoff = df * (p * payoffu + (1-p) * payoffd)
                    payoff_list[j][i].append(payoff)
                if AorE == 'A':
                    for k1 in range(a, len(Smax_list[j][i+1])):
                        if Smax_list[j][i+1][k1] == Smax_list[j][i][k]:
                            payoffu = payoff_list[j][i+1][k1]
                            a = k1
                            index = 1
                        if index == 0:
                            payoffu = payoff_list[j][i + 1][-1]
                    for k2 in range(b, len(Smax_list[j+1][i+1])):
                        if Smax_list[j+1][i+1][k2] == Smax_list[j][i][k]:
                            payoffd = payoff_list[j+1][i+1][k2]
                            b = k2
                    payoff = max(df * (p * payoffu + (1-p) * payoffd), Smax_list[j][i][k]-S[j,i])
                    payoff_list[j][i].append(payoff)
    # print(payoff_list)
    # print(payoff_list[0][0])
    return payoff_list[0][0]


# Eput50 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=50, AorE='E')
# print(f"European puts Smaxt=50: {Eput50}")
# Aput50 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=50, AorE='A')
# print(f"American puts Smaxt=50: {Aput50}")
# Eput60 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=60, AorE='E')
# print(f"European puts Smaxt=60: {Eput60}")
# Aput60 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=60, AorE='A')
# print(f"American puts Smaxt=60: {Aput60}")
# Eput70 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=70, AorE='E')
# print(f"European puts Smaxt=70: {Eput70}")
# Aput70 = lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=70, AorE='A')
# print(f"American puts Smaxt=70: {Aput70}")


def MonteCarlo_lookback_puts(St, r, q, sigma, t, T, Smaxt, n, ns, nr):
    dt = (T - t) / n
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    payoff_mean = []
    for repeat in range(nr):
        payoff_list = []
        # print(f"跑到第{repeat}次")
        for i in range(ns):
            # nd = np.random.normal()
            S = St
            Smax = max(St, Smaxt)
            S_list = []
            for j in range(n):
                nd = np.random.normal()
                S *= np.exp((r - q - 0.5 * sigma**2) * dt + nd * sigma * dt**0.5)
                S_list.append(S)
            payoff = max(np.max(S_list), Smax) - S
            discount_payoff = np.exp(-r * (T-t)) * payoff
            payoff_list.append(discount_payoff)
        payoff_mean.append(np.mean(payoff_list))
    price = np.mean(payoff_mean)
    price_se = np.std(payoff_mean)
    price_down = price - 2 * price_se
    price_up = price + 2 * price_se
    # print(f"Monte Carlo price: {price}")
    # print(f"95%C.I:[{price_down, price_up}]")
    return price  # 在financial algorithm檔案呼叫時，要有return值

# MonteCarlo_lookback_puts(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, Smaxt=50, n=100, ns=10000, nr=20)
# MonteCarlo_lookback_puts(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, Smaxt=60, n=100, ns=10000, nr=20)
# MonteCarlo_lookback_puts(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, Smaxt=70, n=100, ns=10000, nr=20)