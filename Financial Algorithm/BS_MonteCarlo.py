import numpy as np
import scipy.stats as stats


def N(x):
    return stats.norm.cdf(x)


def Black_Scholes(St, K, r, q, sigma, T):
    d1 = (np.log(St/K)+(r-q+0.5*sigma**2)*T) / (sigma * T**0.5)
    d2 = d1 - sigma * (T**0.5)

    C = St * np.exp(-q*T) * N(d1) - K * np.exp(-r*T) * N(d2)
    print("call price:", C)
    P = K * np.exp(-r*T) * N(-d2) - St * np.exp(-q*T) * N(-d1)
    print("put price:", P)


# Black_Scholes(St=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5)


def MonteCarlo(St, K, r, q, sigma, T, callorput, ns, nr):
    C_list = []
    P_list = []
    C_price_list = []
    P_price_list = []
    for j in range(nr):
        for i in range(ns):  # 生出10000次normal亂數
            nd = np.random.normal()
            ST = np.exp(np.log(St)+(r-q-0.5*(sigma**2))*T + T**0.5 * nd * sigma)
            if K < ST:
                C = np.exp(-r*T) * (ST - K)
                P = 0
            elif K > ST:
                C = 0
                P = np.exp(-r*T) * (K - ST)
            else:
                C = P = 0
            C_list.append(C)  # 將每個ST對應的C儲存
            P_list.append(P)
        C_price = np.mean(C_list)  # 算出這10000個C取平均並儲存
        P_price = np.mean(P_list)
        C_price_list.append(C_price)  # 儲存共20次的price
        P_price_list.append(P_price)
    C_mean = np.mean(C_price_list)
    C_se = np.std(C_price_list)
    P_mean = np.mean(P_price_list)
    P_se = np.std(P_price_list)
    if callorput == 'call':
        print(f"Call price: {C_mean}")
        print(f"95%C.I:[{C_mean - 2*C_se}, {C_mean + 2*C_se}]")
        return C_mean
    elif callorput =='put':
        print(f"Put price: {P_mean}")
        print(f"95%C.I:[{P_mean - 2*P_se}, {P_mean + 2*P_se}]")
        return P_mean


# MonteCarlo(St=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, callorput='call', ns=10000, nr=20)
# MonteCarlo(St=50, K=50, r=0.1, q=0.05, sigma=0.4, T=0.5, callorput='put', ns=10000, nr=20)
