import statistics
import numpy as np
import scipy.stats as stats

def N(x):
    return stats.norm.cdf(x)
def ln(x):
    return np.log(x)
def e(x):
    return np.exp(x)
def norm(x):
    return np.random.normal(x)

S0 = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.4
T = 0.5

# S0 = float(input("S0 = "))
# r = float(input("r = "))
# q = float(input("q = "))
# sigma = float(input("sigma = "))
# T = float(input("T = "))
# K = float(input("K = "))

d1 = (ln(S0/K)+(r-q+0.5*sigma**2)*T) / (sigma * T**0.5)
d2 = d1 - sigma * (T**0.5)

C = S0 * e(-q*T) * N(d1) - K * e(-r*T) * N(d2)
print("call price:", C)
P = K * e(-r*T) * N(-d2) - S0 * e(-q*T) * N(-d1)
print("put price:", P)

C_list = []
P_list = []
C_price_list = []
P_price_list = []
for j in range(20):
    for i in range(10000):  # 生出10000次normal亂數
        nd = []
        nd = np.random.normal()
        ST = e(ln(S0)+(r-q-0.5*(sigma**2))*T + T**0.5 * nd * sigma)
        if K < ST:
            C = e(-r*T) * (ST - K)
            P = 0
        elif K > ST:
            C = 0
            P = e(-r*T) * (K - ST)
        else:
            C = P = 0
        C_list.append(C)  # 將每個ST對應的C儲存
        P_list.append(P)
    C_price = statistics.mean(C_list)  # 算出這10000個C取平均並儲存
    P_price = statistics.mean(P_list)
    C_price_list.append(C_price)  # 儲存共20次的price
    P_price_list.append(P_price)

C_mean = statistics.mean(C_price_list)
C_se = statistics.pstdev(C_price_list)
P_mean = statistics.mean(P_price_list)
P_se = statistics.pstdev(P_price_list)

C_down = C_mean - 2 * C_se
C_up = C_mean + 2 * C_se
P_down = P_mean - 2 * P_se
P_up = P_mean + 2 * P_se
print(C_mean)
print(f"95%C.I:[{C_down}, {C_up}]")
print(P_mean)
print(f"95%C.I:[{P_down}, {P_up}]")
