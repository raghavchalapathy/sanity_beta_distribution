import numpy as np
from scipy.optimize import minimize, check_grad

from myOCSVM import *

def hiddenScore(X, V, bH, g):
    return g(X.dot(V) + bH)

def nnScore(X, w, V, bH, g):
    return hiddenScore(X, V, bH, g).dot(w)

def ocnn_frozenvb_obj(theta, X, nu, D, K, V, bH, g, dG):
    
    w = theta[:K]
    r = theta[-1]
    
    term1 = 0.5  * np.sum(w**2)
    term2 = 0
    term3 = 1/nu * np.mean(relu(r - nnScore(X, w, V, bH, g)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

def ocnn_frozenvb_grad(theta, X, nu, D, K, V, bH, g, dG):
    
    N = X.shape[0]
    w = theta[:K]
    r = theta[-1]
    
    deriv = dRelu(r - nnScore(X, w, V, bH, g))

    term1 = np.concatenate(( w,
                             np.zeros((1,)) ))

    term2 = np.concatenate(( np.zeros((w.size,)),
                             np.zeros((1,)) ))

    term3 = np.concatenate(( 1/nu * np.mean(deriv[:,np.newaxis] * (-hiddenScore(X, V, bH, g)), axis = 0),
                             1/nu * np.array([ np.mean(deriv) ]) ))
    
    term4 = np.concatenate(( np.zeros((w.size,)),
                             -1 * np.ones((1,)) ))
    
    return term1 + term2 + term3 + term4


class MyOCNN_FrozenVB:

    def __init__(self, nu, R, bH, g, dG):

        self.nu = nu
        self.R  = R
        self.bH = bH
        self.g  = g
        self.dG = dG

    def fit(self, XTr, theta0 = None):

        D = XTr.shape[1]
        K = self.R.shape[1]

        if theta0 is None:
            theta0 = np.random.normal(0, 1, K + 1)
        #theta0 = resEXP.x

        print('Gradient error: %s' % check_grad(ocnn_frozenvb_obj, ocnn_frozenvb_grad, theta0, XTr, self.nu, D, K, self.R, self.bH, self.g, self.dG))
        resNN = minimize(ocnn_frozenvb_obj, theta0, 
                         jac     = ocnn_frozenvb_grad, 
                         args    = (XTr, self.nu, D, K, self.R, self.bH, self.g, self.dG),
                         method  = 'L-BFGS-B',                
                         options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})
        print('Gradient error: %s' % check_grad(ocnn_frozenvb_obj, ocnn_frozenvb_grad, resNN.x, XTr, self.nu, D, K, self.R, self.bH, self.g, self.dG))

        self.w = resNN.x[:-1]
        self.r = resNN.x[-1]
