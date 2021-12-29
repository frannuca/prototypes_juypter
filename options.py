import numpy as np
from scipy.stats import norm
import datetime as dt
import matplotlib.pyplot as plt

def C_v(S,K,T,r,q,sigma):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: dividend rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    result = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))        
    return result

def P_v(S,K,T,r,q,sigma):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: dividend rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    result = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))        
    return result

def C_d(S,K,T,r,q,sigma):
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r*T)*norm.cdf(d2)

def P_d(S,K,T,r,q,sigma):
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r*T)*norm.cdf(-d2)

#https://people.maths.ox.ac.uk/howison/barriers.pdf

def C_do(S,K,T,r,q,sigma,BL):
    if S<BL:
        return 0.0
        
    s2 = sigma**2
    k = 2*(r-q)/s2
    a = 0.5*(1-k)
    b = -0.25*(k-1)**2-k
    if BL < K:
        cbs = C_v(S,K,T,r,q,sigma)        
        return  cbs - (S/BL)**(2*a)*C_v(BL**2/S,K,T,r,q,sigma)
    else:
    
        return (C_v(S,BL,T,r,q,sigma) + (BL-K)*C_d(S,BL,T,r,q,sigma) 
               -(S/BL)**(2*a)*(C_v(BL**2/S,BL,T,r,q,sigma) + (BL-K)*C_d(BL**2/S,BL,T,r,q,sigma))
               )

 


def C_di(S,K,T,r,q,sigma,BL):
    s2 = sigma**2
    k = 2*(r-q)/s2
    a = 0.5*(1-k)
    b = -0.25*(k-1)**2-k
    return (S/BL)**(2*a)*C_v(BL**2/S,K,T,r,q,sigma)


def P_do(S,K,T,r,q,sigma,BL):
    if S<BL:
        return 0.0

    s2 = sigma**2
    k = 2*(r-q)/s2
    a = 0.5*(1-k)
    b = -0.25*(k-1)**2-k

    return (P_v(S,K,T,r,q,sigma) - P_v(S,BL,T,r,q,sigma)  - (K-BL)*P_d(S,BL,T,r,q,sigma) 
            - (S/BL)**2 * (P_v(BL**2/S,K,T,r,q,sigma) - P_v(BL**2/S,BL,T,r,q,sigma) - (K-BL)*P_d(BL**2/S,BL,T,r,q,sigma) )
            )

def C_uo(S,K,T,r,q,sigma,BL):
    if S>BL:
        return 0.0

    s2 = sigma**2
    k = 2*(r-q)/s2
    a = 0.5*(1-k)
    b = -0.25*(k-1)**2-k

    npv = (C_v(S,K,T,r,q,sigma) - C_v(S,BL,T,r,q,sigma)  - (BL-K)*C_d(S,BL,T,r,q,sigma) 
            - (S/BL)**2 * (C_v(BL**2/S,K,T,r,q,sigma) - C_v(BL**2/S,BL,T,r,q,sigma) - (BL-K)*C_d(BL**2/S,BL,T,r,q,sigma) )
            )
    return npv

def P_uo(S,K,T,r,q,sigma,BL):
    if S>BL:
        return 0.0

    s2 = sigma**2
    k = 2*(r-q)/s2
    a = 0.5*(1-k)
    b = -0.25*(k-1)**2-k
    if BL>K:  
        npv = P_v(S,K,T,r,q,sigma)  - (S/BL)**(2*a)*P_v(BL**2/S,K,T,r,q,sigma)
  
    else:
        npv = (P_v(S,BL,T,r,q,sigma) + (K-BL)*P_d(S,BL,T,r,q,sigma) 
                -(S/BL)**(2*a)*(P_v(BL**2/S,BL,T,r,q,sigma) + (K-BL)*P_d(BL**2/S,BL,T,r,q,sigma))
                )
    return npv

