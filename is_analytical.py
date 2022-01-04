import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from options import *

class OptionInstrument:
    def __init__(self,spot, K, r, q, T, sigma,notional,iscall):
        self.spot=spot
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.T = T
        self.notional = notional
        self.isCall = iscall
        
        if iscall:
            self.npv = lambda s: C_v(S=s,K=self.K,T=self.T,r=self.r,q=self.q,sigma=self.sigma)*self.notional
            self.baseline = C_v(S=self.spot,K=self.K,T=self.T,r=self.r,q=self.q,sigma=self.sigma)*self.notional
        else:
            self.npv = lambda s: P_v(S=s,K=self.K,T=self.T,r=self.r,q=self.q,sigma=self.sigma)*self.notional
            self.baseline = P_v(S=self.spot,K=self.K,T=self.T,r=self.r,q=self.q,sigma=self.sigma)*self.notional

    def loss(self, S):
        S = max(0.0,S)        
        return (self.npv(S) - self.baseline)

        

volfactor = np.sqrt(10.0/260.0)
portfolio= [OptionInstrument(spot=100.0,K=100.0,T=1.00,notional=50,r=0.01,q=0.0,sigma=0.50,iscall=True),
            OptionInstrument(spot=100.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.15,iscall=True),
            OptionInstrument(spot=100.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.25,iscall=True),
            OptionInstrument(spot=100.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.15,iscall=True),
            OptionInstrument(spot=110.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.05,iscall=True),
            OptionInstrument(spot=110.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.45,iscall=True),
            OptionInstrument(spot=110.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.15,iscall=True),
            OptionInstrument(spot=110.0,K=100.0,T=0.75,notional=50,r=0.01,q=0.0,sigma=0.07,iscall=True)
            ]
def compute_portfolio_loss(xshocks):
    return np.sum(p.loss(p.spot*(1+s)) for (p,s) in zip(portfolio,xshocks))

def compute_portfolio_npv(xS):
    return np.sum(p.npv(s) for (p,s) in zip(portfolio,xS))
   

def get_simulation_losses(cube):
    loss=[]
    for i in range(cube.shape[1]):
        xS = np.squeeze(np.asarray(cube[:,i]))
        loss.append(compute_portfolio_loss(xS))
    return np.array(loss)

def get_approx_mu(Cov,cl,x=None):
    np.random.seed(42)
    
    a = 0.0
    S=np.array([p.spot for p in portfolio])
    ds=0.01
    L = np.linalg.linalg.cholesky(Cov)
    d = []
    for i in range(len(portfolio)):        
        Sp = S.copy()
        Sp[i] *= (1+ds)
        xp = compute_portfolio_npv(Sp)
        Sm = S.copy()
        Sm[i] *= (1-ds)
        xm = compute_portfolio_npv(Sm)
        delta = (xp - xm)/(2*ds)        
        d.append(delta)
    
    
    
    d = np.matrix(d).reshape(len(portfolio),1)   
    b = np.matmul(L.transpose(),d)
    v = np.sqrt(np.sum(np.multiply(b,b)))
    
    approx_dist = norm(0,v)
    if x is None:
        x = approx_dist.ppf(cl)
    den = np.matmul(b.transpose(),b)[0,0]
    mu =  (x-a)*b/den
    return mu

def approximate_mu_with_fast_mc(nSims, Cov,cl):
    np.random.seed(42)
    nAssets=sigma.shape[0]
    
    rn = norm.rvs(size=(nAssets,nSims))
    C = np.linalg.cholesky(Cov)
    cube = np.matmul(C,rn)
    loss = get_simulation_losses(cube)

    idx = np.argsort(loss)
    nq = idx[int(float(nSims)*cl)]
    return loss[nq]


def run_mc_is(nAssets, nSims, mu,Cov,cl):
    np.random.seed(42)
    C = np.linalg.cholesky(Cov)
    rn = norm.rvs(size=(nAssets,nSims))
    rn_mu = rn + mu
    cube = np.matmul(C,rn_mu)
    loss = get_simulation_losses(cube)
    
    

    lr = []
    
    g = 0.5*np.matmul(mu.transpose(),mu)
    g = g[0,0]
    for i in range(rn_mu.shape[1]):
        n=rn_mu[:,i]
        h = np.matmul(mu.transpose(),n)
        ff = g-h[0,0]
        lr.append(np.exp(ff)/float(nSims))
    
    lr=np.array(lr)
    

    idx = np.argsort(loss)
    loss = loss[idx][::-1]
    cube = cube[:,idx[::-1]]
    lr = lr[idx][::-1]
    finder = lambda n: np.sum(lr[:n])
    n=0
    alpha = 1.0-cl
    while finder(n)<=alpha and n<len(lr)-1:
        n=n+1
    
    phigh = finder(n)
    plow = finder(n-1)
    lhigh = loss[n-1]
    llow = loss[n]
    VaR=llow + (lhigh-llow)*(alpha-plow)/(phigh-plow)

    return VaR


rho=np.ones((len(portfolio),len(portfolio)))*0.5
for i in range(rho.shape[0]):
    rho[i,i]=1.0

sigma = np.diag(np.array([p.sigma*volfactor for p in portfolio]))
cov=np.matmul(np.matmul(sigma,rho),sigma)
cl = 0.99

nsims_v = []

nSims0 = 500
mc=[]
ms_n=[]
ismc=[]
ismc_update=[]

lossx= approximate_mu_with_fast_mc(50000,cov,cl)
mu = get_approx_mu(cov,cl)

for niter in range(6):
    nsims = nSims0*2**niter
    ms_n.append(nsims)
    print(f"runnin IS MC for {nsims} simulations")    
    loss= run_mc_is(nAssets=len(portfolio),mu = mu, nSims=nsims,Cov=cov,cl=cl)
    ismc.append(loss)
    print(f"VaR={loss}, mu = {mu.transpose()}")

print(f"VaRx MC = {lossx}")
for nsims in ms_n:
    print(f"mu={mu.transpose()}")
    print(f"runnin IS MC Update for {nsims} simulations")    
    loss= run_mc_is(nAssets=len(portfolio),mu = mu, nSims=nsims,Cov=cov,cl=cl)
    mu = get_approx_mu(Cov=cov,cl=cl,x=loss)    
    ismc_update.append(loss)
    print(f"VaR={loss}, mu = {mu.transpose()}")

for nsims in ms_n:   
   print(f"running MC for {nsims} simulations")
   loss= approximate_mu_with_fast_mc(nsims,cov,cl)
   mc.append(loss)
   

print(f"convergence value VaR={lossx}")
plt.plot((ms_n),mc,label="MC")
plt.plot((ms_n),ismc,label="IS MC")
plt.plot((ms_n),ismc_update,label="IS MC Update")
plt.plot(ms_n,[lossx]*len(ms_n),'-.',label="Asymptotic Value")
plt.xlabel("Number of Simulations")
plt.ylabel("Portfolio Losses")
plt.legend()
plt.grid()
plt.show()
