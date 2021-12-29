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
            OptionInstrument(spot=100.0,K=99.0,T=0.5,notional=-50,r=0.01,q=0.0,sigma=0.65,iscall=False),
            OptionInstrument(spot=100.0,K=99.0,T=0.5,notional=-50,r=0.01,q=0.0,sigma=0.25,iscall=False),
            OptionInstrument(spot=100.0,K=99.0,T=0.5,notional=150,r=0.01,q=0.0,sigma=0.25,iscall=False),
            OptionInstrument(spot=100.0,K=105.0,T=0.5,notional=150,r=0.01,q=0.0,sigma=0.25,iscall=True),
            ]
def compute_portfolio_loss(xS):
    return np.sum(p.loss(p.spot*(1+s)) for (p,s) in zip(portfolio,xS))

def compute_portfolio_npv(xS):
    return np.sum(p.npv(s) for (p,s) in zip(portfolio,xS))
   

def get_simulation_losses(cube):
    loss=[]
    for i in range(cube.shape[1]):
        xS = np.squeeze(np.asarray(cube[:,i]))
        loss.append(compute_portfolio_loss(xS))
    return np.array(loss)

def get_approx_mu(Cov,cl):
    np.random.seed(42)
    
    a = 0.0
    S=np.array([p.spot for p in portfolio])

    L = np.linalg.linalg.cholesky(Cov)
    d = []
    for i in range(len(portfolio)):        
        Sp = S.copy()
        Sp[i] *= 1.01
        xp = compute_portfolio_npv(Sp)
        Sm = S.copy()
        Sm[i] *= 0.99
        xm = compute_portfolio_npv(Sm)
        delta = (xp - xm)/(0.01)        
       
        d.append(delta)
    
    
    
    d = np.matrix(d).reshape(len(portfolio),1)   
    b = np.matmul(L.transpose(),d)
    v = np.sqrt(np.sum(np.multiply(b,b)))
    approx_dist = norm(0,v)
    x = approx_dist.ppf(cl)
    den = np.matmul(b.transpose(),b)[0,0]
    mu =  (x-a)*b/den
    return mu
    
def approximate_mu_with_fast_mc(nSims, sigma, rho,cl):
    np.random.seed(42)
    nAssets=sigma.shape[0]
    l = sigma
    Cov = np.matmul(np.matmul(l,rho),l)
    rn = norm.rvs(size=(nAssets,nSims))
    C = np.linalg.cholesky(Cov)
    cube = np.matmul(C,rn)
    loss = get_simulation_losses(cube)

    idx = np.argsort(loss)
    nq = idx[int(float(nSims)*cl)]
    mu = np.matrix(np.squeeze(np.array(cube[:,nq]))).reshape(nAssets,1)
    return Cov,mu,loss[nq]



def run_mc_is(nAssets, nSims, mu,Cov,cl):
    np.random.seed(42)
    rn = norm.rvs(size=(nAssets,nSims)) + mu.transpose()
    C = np.linalg.cholesky(Cov)

    lr = []
    g = 0.5*mu*mu.transpose()
    g = g[0,0]
    for i in range(rn.shape[1]):
        n=rn[:,i]
        h = np.matmul(mu,n)
        ff = g-h[0,0]
        lr.append(np.exp(ff)/float(nSims))
    
    lr=np.array(lr)
    cube = np.matmul(C,rn)
    loss = get_simulation_losses(cube)

    idx = np.argsort(loss)
    loss = loss[idx][::-1]
    cube = cube[:,idx[::-1]]
    lr = lr[idx][::-1]
    finder = lambda n: np.sum(lr[:n])
    n=0
    alpha = 1.0-cl
    while finder(n)<=alpha:
        n=n+1
    
    phigh = finder(n)
    plow = finder(n-1)
    lhigh = loss[n-1]
    llow = loss[n]
    VaR=llow + (lhigh-llow)*(alpha-plow)/(phigh-plow)
    mu_new = np.matrix(np.array(cube[:,n-1]))

    return mu_new,VaR



def run_mc_is_post(nAssets, nSims,Cov,cl):
    np.random.seed(42)

    mu = get_approx_mu(Cov,cl).transpose()
    rn = norm.rvs(size=(nAssets,nSims)) 
    C = np.linalg.cholesky(Cov)
    cube = np.matmul(C,rn)
    loss = get_simulation_losses(cube)

    lr = []
    g = 0.5*mu*mu.transpose()
    g = g[0,0]
    muctr = np.matmul(mu,mu.transpose())*0.5
    for i in range(rn.shape[1]):
        n0=rn[:,i] + mu.transpose()
        mu_z = np.matmul(mu,n0)
        lr.append(np.exp(muctr-mu_z)/float(nSims))
    
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
    mu_new = np.matrix(np.array(cube[:,n-1]))

    return mu_new,VaR


rho=np.diag(np.ones(len(portfolio))*0.5)
for i in range(rho.shape[0]):
    rho[i,i]=1.0

sigma = np.diag(np.array([p.sigma*volfactor for p in portfolio]))

cl = 0.999
cov,mu,approx_loss = approximate_mu_with_fast_mc(int(1.0/(1.0-cl)),sigma,rho,cl)
#cov,mu = get_approx_mu(sigma,rho,approx_loss)
nsims_v = []

ismc = []
isnsims = 1000
isnsamples = []
samplesCounter=0
for niter in range(5):
    print(f"runnin IS MC for {isnsims} simulations")    
    mu,loss= run_mc_is_post(nAssets=len(portfolio), nSims=isnsims,Cov=cov,cl=cl)
    print(f"VaR={loss}, mu = {mu.transpose()}")
    samplesCounter =samplesCounter+isnsims
    isnsamples.append(samplesCounter)
    ismc.append(loss)

#mc=[]
#for nsims in isnsamples:
#    print(f"runnin MC for {nsims} simulations")
#    z,s,loss= approximate_mu_with_fast_mc(nsims,sigma,rho,cl)
#    mc.append(loss)


z,s,loss= approximate_mu_with_fast_mc(20000,sigma,rho,cl)
print(f"target MC={loss}")
#plt.plot(np.log10(isnsamples),mc,label="MC")
plt.plot(np.log10(isnsamples),ismc,label="IS MC")
plt.legend()
plt.grid()
plt.show()
