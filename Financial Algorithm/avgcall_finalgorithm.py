import numpy as np
# from time import process_time
# start = process_time()


def arithmetic_avg_call(St, K, r, q, sigma, t, T, n, Smaxt, AorE):  # T_t = T - t

    dt = T / n
    u = np.exp(sigma * dt**0.5)
    d = 1/u
    p = (np.exp((r-q)*dt)-d) / (u-d)
    df = np.exp(-r * dt)  # discount factor 多這個是因為 np.function無法和float相乘
    af = n * t / T  # when t != 0，參數ave_factor需調整

    Amax_list = np.zeros([n+1, n+1])
    Amin_list = np.zeros([n+1, n+1])
    A_list = [[[] for i in range(n+1)] for j in range(n + 1)]
    # A_list[0][0].append(St)
    payoff_list = [[[] for i in range(n+1)] for j in range(n+1)]

    A = 0
    for i in range(n+1):
        for j in range(i+1):
            Amax_list[j, i] = (Smaxt * (af+1) + St * u * (1 - u**(i-j)) / (1-u) + St * u**(i-j) * d * (1-d**j) / (1-d)) / (i+af+1)
            Amin_list[j, i] = (Smaxt * (af+1) + St * d * (1-d**j) / (1-d) + St * d**j * u * (1-u**(i-j)) / (1-u)) / (i+af+1)
            for k in range(100+1):
                A = np.exp(((100 - k) / 100) * np.log(Amax_list[j, i]) + (k / 100) * np.log(Amin_list[j, i]))
                A_list[j][i].append(A)
                payoff = max(A_list[j][i][k] - K, 0)
                payoff_list[j][i].append(payoff)
    Cu = 0
    Cd = 0
    Cjik = 0
    for i in range(n-1, -1, -1):  # Backward induction
        # print(f"跑到第{i}圈")
        for j in range(i+1):
            indexu = 1
            indexd = 1
            for k in range(100+1):
                Au = ((i+af+1) * A_list[j][i][k] + St * u**(i+1-j) * d**j) / (i+af+2)

                if j == 0:  # 右上角全部股價全部上升u的部分，Amax=Amin，故另外設條件式以免前後相減分母會為0
                    Cu = payoff_list[j][i+1][k]
                else:
                    if Au > A_list[j][i+1][0]:
                        Cu = payoff_list[j][i+1][0]
                    elif Au < A_list[j][i+1][-1]:
                        Cu = payoff_list[j][i+1][-1]
                    else:
                        for lu in range(indexu, 100+1):
                            if A_list[j][i+1][lu] <= Au <= A_list[j][i+1][lu-1]:
                                wu = (A_list[j][i+1][lu-1] - Au) / (A_list[j][i+1][lu-1] - A_list[j][i+1][lu])
                                Cu = wu * payoff_list[j][i+1][lu] + (1-wu) * payoff_list[j][i+1][lu-1]
                                indexu = lu
                                break
                Ad = ((i+af+1) * A_list[j][i][k] + St * u**(i+1-(j+1)) * d**(j+1)) / (i+af+2)

                if i == j:  # 主對角線股價全部下降的部分，Amax = Amin，故另外設條件式以免前後相減分母會為0
                    Cd = payoff_list[j+1][i+1][k]
                else:
                    if Ad > A_list[j+1][i+1][0]:
                        Cd = payoff_list[j+1][i+1][0]
                    elif Ad < A_list[j+1][i+1][-1]:
                        Cd = payoff_list[j+1][i+1][-1]
                    else:
                        for ld in range(indexd, 100+1):
                            if A_list[j+1][i+1][ld] <= Ad <= A_list[j+1][i+1][ld-1]:
                                wd = (A_list[j+1][i+1][ld-1] - Ad) / (A_list[j+1][i+1][ld-1] - A_list[j+1][i+1][ld])
                                Cd = wd * payoff_list[j+1][i+1][ld] + (1-wd) * payoff_list[j+1][i+1][ld-1]
                                indexd = ld
                                break
                if AorE == 'E':
                    Cjik = (p * Cu + (1 - p) * Cd) * df
                elif AorE == 'A':
                    Cjik = max(A_list[j][i][k] - K, (p * Cu + (1 - p) * Cd) * df)
                payoff_list[j][i][k] = Cjik
    return payoff_list[0][0][0]


# E_linear_0 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='E', method = 'linearly')
# print(f"European calls t = 0: {E_linear_0}")
# E_log_0 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='E', method = 'log')
# print(f"European calls t = 0: ({E_log_0})")
# E_linear_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='E', method = 'linearly')
# print(f"European calls t = 0.25: {E_linear_t}")
# E_log_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='E', method = 'log')
# print(f"European calls t = 0.25: ({E_log_t})")
#
# A_linear_0 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'linearly')
# print(f"American calls t = 0: {A_linear_0}")
# A_log_0 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'log')
# print(f"American calls t = 0: ({A_log_0})")
# A_linear_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'linearly')
# print(f"American calls t = 0.25: {A_linear_t}")
# A_log_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'log')
# print(f"American calls t = 0.25: ({A_log_t})")


def MonteCarlo_arithmetic_avg_call(St, K, r, q, sigma, t, T, n, Smaxt, ns, nr):

    dt = T / n
    af = n * t / T
    df = np.exp(-r * dt)  # discount factor 多這個是因為 np.function無法和float相乘

    call_price = []
    for repeat in range(nr):
        # print(f"跑到第{repeat}圈")
        payoff_list = []
        for simulation in range(ns):
            S = St
            S_sum = 0
            for i in range(n):
                nd = np.random.normal()
                S *= np.exp((r - q - 0.5*sigma**2) * dt + nd * sigma * dt**0.5)
                S_sum += S
            S_ave = ((af+1) * Smaxt + S_sum) / (af+n+1)
            payoff = max(S_ave-K, 0)
            payoff_list.append(payoff)
        call_price.append(np.mean(payoff_list) * df)

    mean = np.mean(call_price)
    std = np.std(call_price)
    # print(f"Monte Carlo price: {mean}")
    # print(f"95%C.I:[{mean - 2 * std, mean + 2 * std}]")
    return mean


# MonteCarlo_arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T=0.25, n=100, Smaxt=50, ns=10000, nr=20)
# MonteCarlo_arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, n=100, Smaxt=50, ns=10000, nr=20)

# end = process_time()
# print(f"共花了{end-start}秒")
