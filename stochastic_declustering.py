from sklearn.neighbors import KernelDensity
from scipy.stats import norm, expon
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from functools import partial


"""
Things to fix:
    - Boundary problems
    - Bandwidth selection
"""

class stochastic_declustering:
    def __init__(self,events,T=1,iht=False,tol=5e-2):
        """
        Parameters:
            events (np.array [n,3+m]): point locations (x,y,t) and m marks
        """
        self.events = events
        self.n = len(events)
        self.skde = KernelDensity(kernel='gaussian', bandwidth="silverman")
        self.iht = iht
        if iht:
            self.tkde = KernelDensity(kernel='gaussian', bandwidth="silverman")
        self.T = T
        self.tol = tol
    
    def bg_est(self):
        s = (self.events.T[0].var()**0.5+self.events.T[1].var()**0.5)/2
        self.skde.set_params(bandwidth= self.prob_background.sum()**(-1/6)*s)
        self.skde.fit(self.events[:,:2],sample_weight=self.prob_background)
        s_int = self.skde.score_samples(self.events[:,:2])
        
        if self.iht:
            self.tkde.set_params(bandwidth=self.prob_background.sum()**(-0.2)*self.events.T[2].var()**0.5)
            self.tkde.fit(self.events[:,2:3],sample_weight=self.prob_background)
            t_int = self.tkde.score_samples(self.events[:,2:3])
        else:
            t_int = -np.log(self.T)
        return np.exp(s_int+t_int)*self.prob_background.sum()
    
    def trigger(self,x,X,par):
        """
        Compute Trigger Intenisty
        Parameters:
            x (np.array [3+m]): single evaluation point (x,y,t) + marks
            X (np.array [k,3+m]): history points
            par (np.array [4+m])
        """
        r_0, b, sigma_1, sigma_2 = par[:4]
        return (r_0*expon.pdf(x[2]-X[:,2],scale=b)*
                norm.pdf((x[0]-X[:,0])/sigma_1)/sigma_1*
                norm.pdf((x[1]-X[:,1])/sigma_2)/sigma_2)
    
    def row_trig(self,i,par):
        mask = self.events[:,2]<self.events[i,2]
        row = np.zeros(self.n)
        row[mask] = self.trigger(self.events[i],self.events[mask],par)
        return row
        
    def get_trig_mat(self, par):
        """ Calculate triggering intensity matrix 
        return:
            (np.array [n,n]): matrix with the intensities from the trigger function.
                Entry i,j is the intensity that point i contributed to the intensity at j
        """
        trig = np.array(pool.map(partial(self.row_trig,par=par),np.arange(self.n)))
        return trig
    
    def neg_log_lik(self,par,bg_int):
        par[:4] = abs(par[:4])
        trig = self.get_trig_mat(par)
        return -(np.log(trig.sum(axis=0)+bg_int).sum() - par[0]*self.n)

    def gen_prob_mat(self,trig,bg):
        tot = trig.sum(axis=0)+bg
        return bg/tot, trig/tot[None,:]

    def EM_step(self, par,bg_int):
        start = time.time()
        res = minimize(lambda x: self.neg_log_lik(x,bg_int), par, method='nelder-mead',
                   options={'xatol': 1e-5, 'disp': True})
        end = time.time()
        print(f"MLE time {end-start}")
        
        print('Finished MLE')
        par = res['x']
        par[:4] = abs(par[:4])
        print(par)
        trig = self.get_trig_mat(par)
        prob_background, prob_trig = self.gen_prob_mat(trig, bg_int)
        self.prob_background = prob_background
        bg_int = self.bg_est()
        return par, bg_int
    
    def fit(self,x0,A=[[-3,3],[-3,3]]):
        global pool
        pool = Pool()
        bg_int = self.n/((A[0][1]-A[0][0])*(A[1][1]-A[1][0])*self.T)#uniform intensity to begin with
        par = x0
        par_ = x0 - self.tol
        while sum((par-par_)**2) >= self.tol**2:
            par_ = par
            par, bg_int = self.EM_step(par, bg_int)
            print(f"Change in parameters: {sum((par-par_)**2)}")
        self.parameters = par
        pool.close()
        return par
    
    def plot_background(self, A=[[-3,3],[-3,3]]):
        #credit to stackoverflow answer
        xmin, xmax = A[0]
        ymin, ymax = A[1]

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        intensity = np.exp(self.skde.score_samples(positions.T).T)
        f = np.reshape(intensity,xx.shape)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        # Contour plot
        cset = ax.contour(xx, yy, f, colors='k')
        # Label plot
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('Y1')
        ax.set_ylabel('Y0')

        plt.show()

    
    def intensity(self,x):
        #calculate background intensity
        s_int = self.skde.score_samples(x[:,:2])
        if self.iht:
            t_int = self.tkde.score_samples(x[:,2:3])
        else:
            t_int = -np.log(self.T)
        bg_int = np.exp(s_int+t_int)*self.prob_background.sum()
        
        #calculate triggering intensity
        trig_int = np.array([self.trigger(x[i],
                                          self.events[self.events.T[2]<x[i,2]],
                                          self.parameters).sum() 
                             for i in range(len(x))])
        
        return bg_int+trig_int
    def Lamb_t(self,t):
        r_0 = self.parameters[0]
        b = self.parameters[1]
        if self.iht:
            h = self.tkde.bw_
            bg = np.array([(self.prob_background*norm.cdf((t_-self.events.T[2])/h)).sum()/h for t_ in t])
        else:
            bg = self.prob_background.sum()/self.T*t
        return (bg + 
                np.array([r_0*expon.cdf(
                    t_-self.events[self.events.T[2]<t_,2],scale=b
                ).sum() for t_ in t])
               )
    
    
    def delta(self,x):
        r_0, b, sigma_1, sigma_2 = self.parameters[:4]
        a = self.parameters[4:]
        s = x[:2]
        t = x[2]
        mask = self.events.T[2]<t
        trig = self.trigger(x,self.events[mask],self.parameters)
        p = np.zeros(len(self.parameters))
        p[0] = (trig/r_0).sum()
        p[1] = (trig*((t-self.events[mask,2])/b-1)/b).sum()
        p[2] = (trig*((s[0]-self.events[mask,0])**2/sigma_1**3-1/sigma_1)).sum()
        p[3] = (trig*((s[1]-self.events[mask,1])**2/sigma_2**3-1/sigma_2)).sum()
        for i in range(4,len(self.parameters)):
            p[i] = (trig*self.events[mask,i-1]).sum()
        s_int = self.skde.score_samples(s.reshape(1,-1))
        if self.iht:
            t_int = self.tkde.score_samples(t.reshape(-1,1))
        else:
            t_int = -np.log(self.T)
        bg_int = np.exp(s_int+t_int)*self.prob_background.sum()
        return np.outer(p,p)/(bg_int+trig.sum())**2
    
    def wald(self):
        sigma_inv = np.zeros((len(self.parameters),len(self.parameters)))
        for i in range(self.n):
            sigma_inv += self.delta(self.events[i])
        self.S = np.linalg.inv(sigma_inv)
        return self.S

class stochastic_declustering_marks(stochastic_declustering):
    def trigger(self,x,X,par):
        """
        Compute Trigger Intenisty
        Parameters:
            x (np.array [3+m]): single evaluation point (x,y,t) + marks
            X (np.array [k,3+m]): history points
            par (np.array [4+m])
        """
        r_0, b, sigma_1, sigma_2 = par[:4]
        M = np.exp(np.matmul(X[:,3:],par[4:]))
        return (r_0*M*expon.pdf(x[2]-X[:,2],scale=b)*
                norm.pdf((x[0]-X[:,0])/sigma_1)/sigma_1*
                norm.pdf((x[1]-X[:,1])/sigma_2)/sigma_2)
    
    def neg_log_lik(self,par,bg_int):
        par[:4] = abs(par[:4])
        trig = self.get_trig_mat(par)
        M = np.exp(np.matmul(self.events[:,3:],par[4:]))
        return -(np.log(trig.sum(axis=0)+bg_int).sum() - par[0]*M.sum())
    
    def Lamb_t(self,t):
        r_0 = self.parameters[0]
        b = self.parameters[1]
        if self.iht:
            h = self.tkde.bw_
            bg = np.array([(self.prob_background*norm.cdf((t_-self.events.T[2])/h)).sum()/h for t_ in t])
        else:
            bg = self.prob_background.sum()/self.T*t
        
        trig = np.zeros(self.n)
        for i in range(len(t)):
            mask = self.events.T[2]<t[i]
            M = np.exp(np.matmul(self.events[mask,3:],self.parameters[4:]))
            trig[i] = r_0*(M*expon.cdf(t[i]-self.events[mask,2],scale=b)).sum()
        return (bg+trig)