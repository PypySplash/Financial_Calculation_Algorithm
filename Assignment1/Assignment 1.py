import math
import statistics
import numpy as np
import scipy.stats as stats

def N(x):
    return stats.norm.cdf(x)
def ln(x):
    return math.log(x)
def e(x):
    return math.exp(x)
def norm(x):
    return np.random.normal(x)

S0 = 100
r = 0.05
q = 0.02
sigma = 0.5
T = 0.4
K1 = 90
K2 = 98
K3 = 102
K4 = 110
# K4 = 104

# S0 = float(input("S0 = "))
# r = float(input("r = "))
# q = float(input("q = "))
# sigma = float(input("sigma = "))
# T = float(input("T = "))
# K1 = float(input("K1 = "))
# K2 = float(input("K2 = "))
# K3 = float(input("K3 = "))
# K4 = float(input("K4 = "))

d1 = (ln(S0/K1)+(r-q-0.5*sigma**2)*T)/(sigma * T**0.5)
d1_ = d1 + sigma * T**0.5
d2 = (ln(S0/K2)+(r-q-0.5*sigma**2)*T)/(sigma * T**0.5)
d2_ = d2 + sigma * T**0.5
d3 = (ln(S0/K3)+(r-q-0.5*sigma**2)*T)/(sigma * T**0.5)
d3_ = d3 + sigma * T**0.5
d4 = (ln(S0/K4)+(r-q-0.5*sigma**2)*T)/(sigma * T**0.5)
d4_ = d4 + sigma * T**0.5

C = S0*e(-q * T)*(N(d1_)-N(d2_)) \
    - K1*e(-r*T)*(N(d1)-N(d4)) \
    + K2*e(-r*T)*(N(d2)-N(d4)) \
    + ((K2-K1)*K3*e(-r*T)/(K4-K3))*(N(d3)-N(d4)) \
    - (K2-K1)/(K4-K3)*S0*e(-q*T)*(N(d3_)-N(d4_))
print(C)

C_list = []
price_list = []
for j in range(20):
    # nd = np.random.normal(size=10000)
    for i in range(10000):  # 生出10000次normal亂數
        # nd = []
        nd = np.random.normal()
        ST = e(ln(S0)+(r-q-0.5*(sigma**2))*T + math.sqrt(T) * nd * sigma)
        if K1 <= ST < K2:
            C = e(-r*T)*(ST-K1)
        elif K2 <= ST < K3:
            C = e(-r*T)*(K2-K1)
        elif K3 <= ST < K4:
            C = e(-r*T)*((K2-K1)-(K2-K1)/(K4-K3)*(ST-K3))
        else:
            C = 0
        C_list.append(C)  # 將每個ST對應的C儲存
    price = statistics.mean(C_list)  # 算出這10000個C取平均並儲存
    price_list.append(price)  # 儲存共20次的price

mean_value = statistics.mean(price_list)
se_value = statistics.pstdev(price_list)
# print(price_list)

down = mean_value - 2*se_value
up = mean_value + 2*se_value
print(mean_value)
print(f"95%C.I:[{down}, {up}]")
