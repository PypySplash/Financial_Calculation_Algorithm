import numpy as np
import math
from time import process_time
start = process_time()


def arithmetic_avg_call(St, K, r, q, sigma, t, T_t, M, n, Savet, AorE, method):  # T_t = T - t

    dt = T_t / n
    u = np.exp(sigma * dt**0.5)
    d = 1/u
    p = (np.exp((r-q)*dt)-d) / (u-d)
    df = np.exp(-r * dt)  # discount factor 多這個是因為 np.function無法和float相乘
    af = n * t / T_t  # when t != 0，參數ave_factor需調整

    Amax_list = np.zeros([n+1, n+1])
    Amin_list = np.zeros([n+1, n+1])
    A_list = [[[] for i in range(n+1)] for j in range(n + 1)]
    # A_list[0][0].append(St)
    payoff_list = [[[] for i in range(n+1)] for j in range(n+1)]

    A = 0
    for i in range(n+1):
        for j in range(i+1):
            Amax_list[j, i] = (Savet * (af+1) + St * u * (1 - u**(i-j)) / (1-u) + St * u**(i-j) * d * (1-d**j) / (1-d)) / (i+af+1)
            Amin_list[j, i] = (Savet * (af+1) + St * d * (1-d**j) / (1-d) + St * d**j * u * (1-u**(i-j)) / (1-u)) / (i+af+1)
            for k in range(M+1):
                if method == 'linearly':  # basic requirement
                    A = ((M-k) / M) * Amax_list[j, i] + (k/M) * Amin_list[j, i]
                elif method == 'log':  # bonus 1
                    A = np.exp(((M - k) / M) * np.log(Amax_list[j, i]) + (k / M) * np.log(Amin_list[j, i]))
                A_list[j][i].append(A)
                payoff = max(A_list[j][i][k] - K, 0)
                payoff_list[j][i].append(payoff)
    Cu = 0
    Cd = 0
    Cjik = 0
    for i in range(n-1, -1, -1):  # Backward induction
        # print(f"跑到第{i}圈")
        for j in range(i+1):
            # indexu = 1
            # indexd = 1
            u_k = math.floor(M/2)  # 找Au的中點
            d_k = math.floor(M/2)  # 找Ad的中點
            u_limu = 0  # Au的上界
            d_limu = 0  # Ad的上界
            u_limd = M  # Au的下界
            d_limd = M  # Ad的下界
            for k in range(M+1):
                Au = ((i+af+1) * A_list[j][i][k] + St * u**(i+1-j) * d**j) / (i+af+2)

                if j == 0:  # 右上角全部股價全部上升u的部分，Amax=Amin，故另外設條件式以免前後相減分母會為0
                    Cu = payoff_list[j][i+1][k]
                else:
                    if Au >= A_list[j][i+1][0]:
                        Cu = payoff_list[j][i+1][0]
                    elif Au <= A_list[j][i+1][-1]:
                        Cu = payoff_list[j][i+1][-1]
                    else:  # binary search
                        while True:
                            if A_list[j][i+1][u_k] > Au:
                                if abs(u_limd - u_k) <= 1:
                                    wu = (A_list[j][i+1][u_limd-1] - Au) / (A_list[j][i+1][u_limd-1] - A_list[j][i+1][u_limd])
                                    Cu = wu * payoff_list[j][i+1][u_limd] + (1-wu) * payoff_list[j][i+1][u_limd-1]
                                    break
                                u_limu = u_k
                                u_k = round((u_limu + u_limd) / 2)
                            elif A_list[j][i+1][u_k] < Au:
                                if abs(u_k - u_limu) <= 1:
                                    wu = (A_list[j][i+1][u_k-1] - Au) / (A_list[j][i+1][u_k-1] - A_list[j][i+1][u_k])
                                    Cu = wu * payoff_list[j][i+1][u_k] + (1-wu) * payoff_list[j][i+1][u_k-1]
                                    break
                                u_limd = u_k
                                u_k = round((u_limu + u_limd) / 2)
                            elif A_list[j][i+1][u_k] == Au:
                                Cu = payoff_list[j][i+1][u_k]
                                break
                Ad = ((i+af+1) * A_list[j][i][k] + St * u**(i+1-(j+1)) * d**(j+1)) / (i+af+2)

                if i == j:  # 主對角線股價全部下降的部分，Amax = Amin，故另外設條件式以免前後相減分母會為0
                    Cd = payoff_list[j+1][i+1][k]
                else:
                    if Ad >= A_list[j+1][i+1][0]:
                        Cd = payoff_list[j+1][i+1][0]
                    elif Ad <= A_list[j+1][i+1][-1]:
                        Cd = payoff_list[j+1][i+1][-1]
                    else:  # Binary search
                        while True:
                            if A_list[j+1][i+1][d_k] > Ad:
                                if abs(d_limd - d_k) <= 1:
                                    wd = (A_list[j+1][i+1][d_limd-1] - Ad) / (A_list[j+1][i+1][d_limd-1] - A_list[j+1][i+1][d_limd])
                                    Cd = wd * payoff_list[j+1][i+1][d_limd] + (1-wd) * payoff_list[j+1][i+1][d_limd-1]
                                    break
                                d_limu = d_k
                                d_k = round((d_limu + d_limd) / 2)
                            elif A_list[j+1][i+1][d_k] < Ad:
                                if abs(d_k - d_limu) <= 1:
                                    wd = (A_list[j+1][i+1][d_k-1] - Ad) / (A_list[j+1][i+1][d_k-1] - A_list[j+1][i+1][d_k])
                                    Cd = wd * payoff_list[j+1][i+1][d_k] + (1-wd) * payoff_list[j+1][i+1][d_k-1]
                                    break
                                d_limd = d_k
                                d_k = round((d_limu + d_limd) / 2)
                            elif A_list[j+1][i+1][d_k] == Ad:
                                Cd = payoff_list[j+1][i+1][d_k]
                                break
                if AorE == 'E':
                    Cjik = (p * Cu + (1 - p) * Cd) * df
                elif AorE == 'A':
                    Cjik = max(A_list[j][i][k] - K, (p * Cu + (1 - p) * Cd) * df)
                payoff_list[j][i][k] = Cjik
    return payoff_list[0][0][0]


E_linear_0 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='E', method = 'linearly')
print(f"European calls t = 0: {E_linear_0}")
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
# print(f"European calls t = 0: ({A_log_0})")
# A_linear_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'linearly')
# print(f"American calls t = 0.25: {A_linear_t}")
# A_log_t = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='A', method = 'log')
# print(f"European calls t = 0.25: ({A_log_t})")

end = process_time()
print(f"共花了{end-start}秒")
