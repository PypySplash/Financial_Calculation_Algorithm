import numpy as np
from numpy.linalg import inv


def Least_Square_Montecarlo(S, K, r, q, sigma, t, T, n, Smax, Save, ns, nr):
    dt = (T-t) / n

    # 建立Stock price array
    S_array = np.zeros(shape=(ns, n+1))
    S_array[:, 0] = S
    # 建立Normal亂數array
    random_array = np.random.normal(0, 1, size=(ns, n))

    for i in range(1, n+1):
        S_array[:, i] = np.exp(np.log(S_array[:, i-1]) + (r-q-0.5*sigma**2) * dt + (sigma*dt**0.5)*random_array[:, i-1])
    payoff_array = np.zeros(shape=(ns, n+1))
    HV_array = np.zeros(shape=(ns, n+1))
    EV_array = np.zeros(shape=(ns, n+1))
    EHV_array = np.zeros(shape=(ns, n+1))

    def regress_vanilla(x, y):
        X = np.vstack([x ** 2, x, np.ones(x.shape)]).T
        x_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
        return x_hat

    def regress_lookback(x, smax, y):
        X = np.vstack([x ** 2, x, smax ** 2, smax, x * smax, np.ones(x.shape)]).T
        x_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
        return x_hat

    # step 1 : determine the payoff for each path at maturity
    # 用講義範例i = 3來想
    payoff_array[:, n] = np.where(K > S_array[:, n], K - S_array[:, n], 0)  # put t = 3 (K-S)
    for i in range(n, 1, -1):  # when i = 3
        # step 2 : determine the payoff for each path at maturity
        EV_array[:, i - 1] = np.where(K > S_array[:, i - 1], K - S_array[:, i - 1], 0)  # t = 3  S > K out of money 沒有holding value, 設為0
        HV_array[:, i - 1] = np.where(payoff_array[:, i] > 0, payoff_array[:, i] * np.exp(-r * dt), 0)  # 折現回去算t = 2 的 holding value , t = 2  S > K out of money 沒有holding value, 設為0
        # 取得 regress的參數
        parameter1 = regress_vanilla(S_array[:, i - 1][EV_array[:, i - 1] > 0], HV_array[:, i - 1][EV_array[:, i - 1] > 0])
        p1 = np.poly1d(parameter1)
        EHV_array[:, i - 1] = np.where(K > S_array[:, i - 1], p1(S_array[:, i - 1]), 0)
        # parameter2 = regress_lookback(S_array[:, i - 1][EV_array[:, i - 1] > 0], Smax, HV_array[:, i - 1][EV_array[:, i - 1] > 0])
        # p2 = np.poly1d(parameter2)
        # EHV_array[:, i - 1] = np.where(K > S_array[:, i - 1], p2(S_array[:, i - 1]), 0)
        payoff_array[:, i - 1] = np.where(EV_array[:, i - 1] > EHV_array[:, i - 1], EV_array[:, i - 1], HV_array[:, i - 1])
    payoff_array[:, 0] = payoff_array[:, 1] * np.exp(-r * dt)
    option_value = np.mean(payoff_array[:, 0])
    return option_value


print(f"plain vanilla put: {Least_Square_Montecarlo(S=50, K=50, r=0.1, q=0.05, sigma=0.4, t=0, T=0.5, n=100, Smax=60, Save=50, ns=10000, nr=20)}")
# print(f"lookback put: {Least_Square_Montecarlo(S=50, K=50, r=0.1, q=0, sigma=0.4, t=0, T=0.25, n=100, Smax=60, Save=50, ns=10000, nr=20)}")
