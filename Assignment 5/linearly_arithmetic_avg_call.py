import numpy as np
from time import process_time
start = process_time()


def arithmetic_avg_call(St, K, r, q, sigma, t, T_t, M, n, Savet, AorE):  # T_t = T - t

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

    for i in range(n+1):
        for j in range(i+1):
            Amax_list[j, i] = (Savet * (af+1) + St * u * (1 - u**(i-j)) / (1-u) + St * u**(i-j) * d * (1-d**j) / (1-d)) / (i+af+1)
            Amin_list[j, i] = (Savet * (af+1) + St * d * (1-d**j) / (1-d) + St * d**j * u * (1-u**(i-j)) / (1-u)) / (i+af+1)
            for k in range(M+1):
                A = ((M-k) / M) * Amax_list[j, i] + (k/M) * Amin_list[j, i]
                A_list[j][i].append(A)
                payoff = max(A_list[j][i][k] - K, 0)
                payoff_list[j][i].append(payoff)
    Cu = 0
    Cd = 0
    Cjik = 0
    for i in range(n-1, -1, -1):
        # print(f"跑到第{i}圈")
        for j in range(i+1):
            indexu = 1
            indexd = 1
            for k in range(M+1):
                Au = ((i+af+1) * A_list[j][i][k] + St * u**(i+1-j) * d**j) / (i+af+2)

                if j == 0:  # 右上角全部股價全部上升u的部分，Amax=Amin，故另外設條件式以免前後相減分母會為0
                    Cu = payoff_list[j][i+1][k]
                else:
                    if Au > A_list[j][i+1][0]:
                        Cu = payoff_list[j][i+1][0]
                    elif Au < A_list[j][i+1][-1]:
                        Cu = payoff_list[j][i+1][-1]
                    else:
                        for lu in range(indexu, M+1):
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
                        for ld in range(indexd, M+1):
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


# Ecall0_050 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=50, n=100, Savet=50, AorE='E')
# print(f"M = 050: {Ecall0_050}")
# Ecall0_100 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='E')
# print(f"M = 100: {Ecall0_100}")
# Ecall0_150 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=150, n=100, Savet=50, AorE='E')
# print(f"M = 150: {Ecall0_150}")
# Ecall0_200 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=200, n=100, Savet=50, AorE='E')
# print(f"M = 200: {Ecall0_200}")
# Ecall0_250 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=250, n=100, Savet=50, AorE='E')
# print(f"M = 250: {Ecall0_250}")
# Ecall0_300 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=300, n=100, Savet=50, AorE='E')
# print(f"M = 300: {Ecall0_300}")
# Ecall0_350 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=350, n=100, Savet=50, AorE='E')
# print(f"M = 350: {Ecall0_350}")
# Ecall0_400 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=400, n=100, Savet=50, AorE='E')
# print(f"M = 400: {Ecall0_400}")

# Ecallt_050 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=50, n=100, Savet=50, AorE='E')
# print(f"M = 050: {Ecallt_050}")
# Ecallt_100 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='E')
# print(f"M = 100: {Ecallt_100}")
# Ecallt_150 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=150, n=100, Savet=50, AorE='E')
# print(f"M = 150: {Ecallt_150}")
# Ecallt_200 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=200, n=100, Savet=50, AorE='E')
# print(f"M = 200: {Ecallt_200}")
# Ecallt_250 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=250, n=100, Savet=50, AorE='E')
# print(f"M = 250: {Ecallt_250}")
# Ecallt_300 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=300, n=100, Savet=50, AorE='E')
# print(f"M = 300: {Ecallt_300}")
# Ecallt_350 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=350, n=100, Savet=50, AorE='E')
# print(f"M = 350: {Ecallt_350}")
# Ecallt_400 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=400, n=100, Savet=50, AorE='E')
# print(f"M = 400: {Ecallt_400}")

# Acall0_050 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=50, n=100, Savet=50, AorE='A')
# print(f"M = 050: {Acall0_050}")
# Acall0_100 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=100, n=100, Savet=50, AorE='A')
# print(f"M = 100: {Acall0_100}")
# Acall0_150 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=150, n=100, Savet=50, AorE='A')
# print(f"M = 150: {Acall0_150}")
# Acall0_200 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=200, n=100, Savet=50, AorE='A')
# print(f"M = 200: {Acall0_200}")
# Acall0_250 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=250, n=100, Savet=50, AorE='A')
# print(f"M = 250: {Acall0_250}")
# Acall0_300 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=300, n=100, Savet=50, AorE='A')
# print(f"M = 300: {Acall0_300}")
# Acall0_350 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=350, n=100, Savet=50, AorE='A')
# print(f"M = 350: {Acall0_350}")
# Acall0_400 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0, T_t=0.25, M=400, n=100, Savet=50, AorE='A')
# print(f"M = 400: {Acall0_400}")

Acallt_050 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=50, n=100, Savet=50, AorE='A')
print(f"M = 050: {Acallt_050}")
Acallt_100 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=100, n=100, Savet=50, AorE='A')
print(f"M = 100: {Acallt_100}")
Acallt_150 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=150, n=100, Savet=50, AorE='A')
print(f"M = 150: {Acallt_150}")
Acallt_200 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=200, n=100, Savet=50, AorE='A')
print(f"M = 200: {Acallt_200}")
Acallt_250 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=250, n=100, Savet=50, AorE='A')
print(f"M = 250: {Acallt_250}")
Acallt_300 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=300, n=100, Savet=50, AorE='A')
print(f"M = 300: {Acallt_300}")
Acallt_350 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=350, n=100, Savet=50, AorE='A')
print(f"M = 350: {Acallt_350}")
Acallt_400 = arithmetic_avg_call(St=50, K=50, r=0.1, q=0.05, sigma=0.8, t=0.25, T_t=0.25, M=400, n=100, Savet=50, AorE='A')
print(f"M = 400: {Acallt_400}")

end = process_time()
print(f"共花了{end-start}秒")
