import numpy as np
import pandas as pd
from scipy.stats import norm
import  matplotlib.pyplot as plt


#P(S <= BL)
def p_hit(T,sigma,S,BL, mu):
    H = BL/S
    x = np.log(H)-mu*T
    sT = sigma*np.sqrt(T)
    return norm.cdf(x/sT)

def p_hit_before_time(T,sigma,S,K, mu):    
    H = K/S    
    if H>1:
        H = 1.0/H
        mu=-mu
        tmp = S
        S=K
        K=tmp
        
    lnK = np.log(H)
    s2 = sigma**2
    sqrtT = np.sqrt(T)
    sigma05T = sigma*sqrtT
    num = np.exp(2*mu*lnK/s2)*norm.cdf((lnK+mu*T)/sigma05T)
    den = norm.cdf((lnK-mu*T)/sigma05T)+1e-12
    return (1+num/(den))*p_hit(T,sigma,S,K,mu)


def p_hit_in_t(T,sigma,S,K, mu):
    H = K/S    
    if H>1:
        H = 1.0/H
        mu=-mu
        tmp = S
        S=K
        K=tmp
        
    dt = 1.0/260.0
    a = p_hit_before_time(T+dt,sigma,S,K, mu)
    b = p_hit_before_time(T-dt,sigma,S,K, mu)

    return (a-b)/(2*dt)


def accumulatorpayoff(S,days_to_maturity,gearing, BL, shares_per_day,sigma,mu):
    tv = np.array(range(1,days_to_maturity))/260.0
    acc_shares = 0.0
    for t in tv:
        acc_shares += shares_per_day*(gearing* p_hit_in_t(t,sigma,S,BL,mu))
    return acc_shares

sigma = 0.25/np.sqrt(260.0)
t = np.array(range(1,3*260))/260.0
dt = t[1]-t[0]
S=4509.37
K=4283.9
shares_per_day = 200
gearing = 1.0
BL_KO=4734.8
mu=0.0
VaR_factor = 0.35
Sup = S*(1+VaR_factor)
Sdown = S*(1-VaR_factor)

base_line = np.cumsum([shares_per_day*(1+gearing*p_hit(i,sigma,S,K,0.0))*(1.0-p_hit_before_time(i,sigma,S,BL_KO, mu))  for i in t])