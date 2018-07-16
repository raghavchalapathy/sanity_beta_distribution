import numpy as np
from scipy.optimize import minimize, check_grad

from myOCNN_FrozenVB import *

def nnProperObj(theta, XPos, XNeg, nu, gam, mu, D, K, g, dG):

    w  = theta[:K]
    V  = theta[K:K+K*D].reshape((D, K))
    bH = theta[K+K*D:K+K*D+K]
    r  = theta[-1]

    term1 = np.maximum(r - nnScore(XPos, w, V, bH, g), 0)
    term2 = mu * 0.5 * (nnScore(XNeg, w, V, bH, g))**2
    term3 = -nu * r

    return np.mean(term1) + np.mean(term2) + term3 + (gam/2) * np.sum(w**2)


def nnProperGrad(theta, XPos, XNeg, nu, gam, mu, D, K, g, dG):

    w = theta[:-1][:,np.newaxis]
    r = theta[-1]

    term1  = r - nnScore(XPos, w, V, bH, g)
    gradH  = (term1 > 0).astype(int) + 0.0 * (term1 == 0).astype(int)
    
    gradW1 = gradH * (-XPos)
    gradR1 = gradH * 1
    
    gradW2 = mu * nnScore(XNeg, w, V, bH, g) * XNeg
    gradR2 = np.array([0])

    gradW3 = np.array([0])
    gradR3 = np.array([-nu])

    gradW = np.mean(gradW1,axis = 0).flatten() + \
            np.mean(gradW2,axis = 0).flatten() + \
            gradW3 + \
            gam * w.flatten()
    gradR = np.mean(gradR1) + \
            np.mean(gradR2) + \
            gradR3

    grad  = np.concatenate([ gradW.flatten(), gradR.flatten() ])

    return grad


class MyNNBGContraster:

    def __init__(self, nu, K, g, dG):

        self.nu = nu
        self.K  = K
        self.g  = g
        self.dG = dG        


    def fit(self, XTr, XBG, theta0):

        D = XTr.shape[1]
        K = self.K

        res = minimize(lambda w : nnProperObj(w, XTr, XBG, self.nu, 1e-2, 1, D, K, self.g, self.dG), 
                       theta0,
                       #jac = lambda w : nnProperGrad(w, XTr, XBG, self.nu, 1e-2, 1, D, K, self.g, self.dG),
                       method = 'nelder-mead',
                       tol = 1e-16)

        self.w = res.x[:K]
        self.V = res.x[K:K+K*D].reshape((D,K))
        self.b = res.x[K+K*D:K+K*D+K]
        self.r = res.x[-1]
