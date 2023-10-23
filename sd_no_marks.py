from scipy.spatial.distance import cdist
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
    def __init__(self,events,T=1,tol=5e-2):
        """
        Parameters:
            events (np.array [n,3]): point locations (x,y,t)
        """
        self.events = events
        self.n = len(events)
        self.kde = KernelDensity(kernel='gaussian', bandwidth="silverman")
        self.T = T
        self.tol = tol

    def bg_est(self):
        s = (self.events.T[0].var()**0.5+self.events.T[1].var()**0.5)/2
        self.kde.set_params(bandwidth= self.n**(-1/6)*s)#prob_background.sum()
        self.kde.fit(self.events[:,:2],sample_weight=self.prob_background)
        return np.exp(self.kde.score_samples(self.events[:,:2]))*self.prob_background.sum()/self.T
    
    def trigger(self,x,X,par):
        """
        Compute Trigger Intenisty
        Parameters:
            x (np.array [3+m]): single evaluation point (x,y,t) + marks
            X (np.array [k,3+m]): history points
            par (np.array [4+m])
        """
        r_0, b, sigma_1, sigma_2 = par
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
        #trig = np.zeros((self.n,self.n))
        #for i in range(self.n):
        #    mask = self.events[:,2]<self.events[i,2]
        #    trig[mask,i] = self.trigger(self.events[i],self.events[mask],par)
        return trig
    
    def neg_log_lik(self,par,bg_int):
        par = abs(par)
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
        par = abs(res['x'])
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
        intensity = np.exp(self.kde.score_samples(positions.T).T)
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
        bg_int = np.exp(self.kde.score_samples(x[:,:2]))*self.prob_background.sum()/self.T
        
        #calculate triggering intensity
        trig_int = np.array([self.trigger(x[i],
                                          self.events[self.events.T[2]<x[i,2]],
                                          self.parameters).sum() 
                             for i in range(len(x))])
        
        return bg_int+trig_int
    
    
    def delta(self,x):
        r_0, b, sigma_1, sigma_2 = self.parameters
        s = x[:2]
        t = x[2]
        mask = self.events.T[2]<t
        trig = self.trigger(x,self.events[mask],self.parameters)
        p = np.zeros(len(self.parameters))
        p[0] = (trig/r_0).sum()
        p[1] = (trig*((t-self.events[mask,2])/b-1)/b).sum()
        p[2] = (trig*((s[0]-self.events[mask,0])**2/sigma_1**3-1/sigma_1)).sum()
        p[3] = (trig*((s[1]-self.events[mask,1])**2/sigma_2**3-1/sigma_2)).sum()
        bg_int = np.exp(self.kde.score_samples([s]))[0]*self.prob_background.sum()/self.T
        return np.outer(p,p)/(bg_int+trig.sum())**2
    
    def wald(self):
        sigma_inv = np.zeros((len(self.parameters),len(self.parameters)))
        for i in range(self.n):
            sigma_inv += self.delta(self.events[i])
        sigma = np.linalg.inv(sigma_inv)
        return sigma
    
    def simulate(self):
        bg = self.kde.sample(n_samples=np.random.poisson(self.prob_background.sum()))
        t = np.random.uniform(size=len(bg))*self.T
        bg = np.concatenate((bg,np.expand_dims(t,-1)),axis=1)
        i = 0
        r_0, b, sigma_1, sigma_2 = self.parameters
        while i < len(bg):
            for j in range(np.random.poisson(lam=r_0)):
                bg = np.concatenate((bg,[bg[i]+[np.random.normal(scale=sigma_1), 
                                               np.random.normal(scale=sigma_2), 
                                               np.random.exponential(b)]]))
            i += 1
        return bg
    
    def super_thinning(self,k,A):
        area = (A[0][1]-A[0][0])*(A[1][1]-A[1][0])
        thinning_mask = np.random.uniform(size=self.n) < (k/self.intensity(self.events))
        #homo_k = gen pois k
        homo_n = np.random.poisson(area*self.T*k)
        homo_k = np.stack((np.random.uniform(size=homo_n)*(A[0][1]-A[0][0])+A[0][0],
                           np.random.uniform(size=homo_n)*(A[1][1]-A[1][0])+A[1][0],
                           np.random.uniform(size=homo_n)*self.T),axis=1)

        homo_mask = np.random.uniform(size=homo_n) < ((k - self.intensity(homo_k))/k)
        inhomo_k = homo_k[homo_mask]

        super_thinned = np.concatenate((self.events[thinning_mask],homo_k[homo_mask]),axis=0)
        print(f"Number of simulated points, {homo_mask.sum()}.\n Number of data points {thinning_mask.sum()}")
        return super_thinned
    
    def Lamb_t(self,t):
        r_0 = self.parameters[0]
        b = self.parameters[1]
        return (self.prob_background.sum()/self.T*t + 
                np.array([r_0*expon.cdf(
                    t_-self.events[self.events.T[2]<t_,2],scale=b
                ).sum() for t_ in t])
               )



def csr_mnn_test(super_thinned,A):
    #MEAN NEAREST NEIGHBORS TEST for complete spatial randomness
    #translate points to [0,1]x[0,1] square
    x = ((super_thinned[:,:2]-np.array([A[0][0],A[1][0]])[None,:])/
         np.array([A[0][1]-A[0][0],A[1][1]-A[1][0]])[None,:])
    #calculate nearest neighbors
    D = cdist(x,x)
    np.fill_diagonal(D, np.inf)
    nearest_neighbors = D.min(axis=0)
    
    # compute mnn statistic
    n = len(nearest_neighbors)
    mu = 0.5*n**(-0.5)+0.206/n+0.164*n**(-3/2)
    sigma = 0.070*n**(-2)+0.148*n**(-5/2)
    z = (nearest_neighbors.mean()-mu)/sigma**(0.5)
    p = 2*norm.cdf(-abs(z))
    print("Mean Nearest Neighbor Complete Spatial Randomness Test on Super Thinned Points")
    print(f"Z-value: {z}, P-value: {p}")
    return z,p