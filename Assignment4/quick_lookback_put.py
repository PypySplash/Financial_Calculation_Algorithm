import numpy as np
from time import process_time
start = process_time()

def quick_lookback_put(St, r, q, sigma, t, T, n, Smaxt, AorE):
    dt = (T - t) / n
    u = np.exp(sigma * dt ** 0.5)
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    # discount factor 多這個是因為 np.function無法和float相乘
    df = np.exp(-r * dt)

    # stock price
    S = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(i + 1):
            S[j, i] = St * d ** j * u ** (i - j)
    # 將股價為50的地方對齊，才不會運算時出現50.00000000001這種浮點數的誤差
    if (n + 1) % 2 == 0:
        S[int((n - 1) / 2), n - 1] = S[0, 0]
    if (n + 1) % 2 != 0:
        S[int(n / 2), n] = S[0, 0]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            S[j][i] = S[j + 1][i + 2]
    # print(S[:,:])
    Smax_list = [[[] for i in range(n + 1)] for j in range(n + 1)]
    Smax_list[0][0].append(max(St, Smaxt))
    for i in range(n + 1):
        num = int((i + 2) / 2)
        for j in range(i + 1):
            if i == 0 & j == 0:  # 原點用股價與Smaxt取大
                continue
            if j == i:
                Smax_list[j][i].append(Smax_list[0][0][0])  # 主對角線上，Smax皆和原點一樣
            if j < i:
                if j == 0:
                    Smax_list[j][i].append(max(S[j,i], Smaxt))  # 水平軸上，Smax為那點的股價與Smaxt取大
                if j != 0:  # quick way mehtod
                    up = i - j  # 股價往上走的次數(u發生的次數)
                    down = j    # 股價向下跌的次數(d發生的次數)
                    num = min(up, down)
                    for k in range(num + 1):  # 每個點的Smax_list裡面Smax的個數
                        if St * u ** (up - k) >= Smaxt:  # 若經過的那條路徑的Smax大於Smaxt，則append那條路徑的Smax
                            Smax_list[j][i].append(St * u ** (up - k))
                        else:                            # 若無則append Smaxt
                            Smax_list[j][i].append(Smaxt)
    for i in range(n + 1):
        for j in range(i + 1):
            for k in range(len(Smax_list[j][i])):
                if Smax_list[j][i][k] - Smaxt < 0.00001:
                    Smax_list[j][i][k] = Smaxt
    # print(Smax_list)
    # 建立一個payoff的三層list
    payoff_list = [[[] for i in range(n + 1)] for j in range(n + 1)]
    for j in range(n + 1):
        for k in range(len(Smax_list[j][n])):  # backward induction
            if Smax_list[j][n][k] >= S[j][n]:
                payoff = Smax_list[j][n][k] - S[j][n]
                if payoff < 10 ** (-5):
                    payoff_list[j][n].append(0)
                else:
                    payoff_list[j][n].append(payoff)
            else:
                payoff = S[j][n] - Smax_list[j][n][k]
                if payoff < 10 ** (-5):
                    payoff_list[j][n].append(0)
                else:
                    payoff_list[j][n].append(payoff)
    for i in range(n+1):
        for j in range(i+1):
            for k in range(len(payoff_list[j][i])):
                if payoff_list[j][i][k] < 10 ** (-5):
                    payoff_list[j][i][k] == 0.0
    # print(payoff_list)
    payoffu = -1
    payoffd = -1
    for i in range(n - 1, -1, -1):
        # print(f"跑到第{i}圈")
        for j in range(i + 1):
            a = 0
            b = 0
            index = 0
            for k in range(len(Smax_list[j][i])):
                if AorE == 'E':
                    for k1 in range(a, len(Smax_list[j][i + 1])):
                        if abs(Smax_list[j][i + 1][k1] - Smax_list[j][i][k]) < 0.00001:
                            payoffu = payoff_list[j][i + 1][k1]
                            a = k1
                            index = 1
                        if index == 0:
                            payoffu = payoff_list[j][i + 1][-1]
                    for k2 in range(b, len(Smax_list[j + 1][i + 1])):
                        if abs(Smax_list[j + 1][i + 1][k2] - Smax_list[j][i][k]) < 0.00001:
                            payoffd = payoff_list[j + 1][i + 1][k2]
                            b = k2
                            index = 1
                    payoff = df * (p * payoffu + (1 - p) * payoffd)
                    payoff_list[j][i].append(payoff)
                if AorE == 'A':
                    for k1 in range(a, len(Smax_list[j][i + 1])):
                        if abs(Smax_list[j][i + 1][k1] -  Smax_list[j][i][k]) < 0.00000001:
                            payoffu = payoff_list[j][i + 1][k1]
                            a = k1
                            index = 1
                        if index == 0:
                            payoffu = payoff_list[j][i + 1][-1]
                    for k2 in range(b, len(Smax_list[j + 1][i + 1])):
                        if abs(Smax_list[j + 1][i + 1][k2] - Smax_list[j][i][k]) < 0.000000001:
                            payoffd = payoff_list[j + 1][i + 1][k2]
                            b = k2
                    payoff = max(df * (p * payoffu + (1 - p) * payoffd), Smax_list[j][i][k] - S[j, i])
                    payoff_list[j][i].append(payoff)
    # print(payoff_list)
    return payoff_list[0][0]

Eput50 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=50, AorE='E')
print(f"European puts Smaxt=50: {Eput50}")
Aput50 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=50, AorE='A')
print(f"American puts Smaxt=50: {Aput50}")
Eput60 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=60, AorE='E')
print(f"European puts Smaxt=60: {Eput60}")
Aput60 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=60, AorE='A')
print(f"American puts Smaxt=60: {Aput60}")
Eput70 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=70, AorE='E')
print(f"European puts Smaxt=70: {Eput70}")
Aput70 = quick_lookback_put(St=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smaxt=70, AorE='A')
print(f"American puts Smaxt=70: {Aput70}")

end = process_time()
print(end-start)