import numpy as np
from scipy.optimize import minimize, check_grad

from myOCNN_FrozenVB import *

def ocfnn_obj(theta, X, nu, b, D, K, g, dG):
    
    w  = theta[:K]
    V  = theta[K:K+K*D].reshape((D, K))
    bH = b #theta[K+K*D:K+K*D+K]
    r  = theta[-1]
    
    term1 = 0.5  * np.sum(w**2)
    term2 = 0 #0.5  * np.sum(V**2)
    term3 = 1/nu * np.mean(relu(r - nnScore(X, w, V, bH, g)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

def ocfnn_grad(theta, X, nu, b, D, K, g, dG):
    
    N  = X.shape[0]
    w  = theta[:K]
    V  = theta[K:K+K*D].reshape((D, K))
    bH = b #theta[K+K*D:K+K*D+K]
    r  = theta[-1]
    
    deriv = dRelu(r - nnScore(X, w, V, bH, g))    

    term1 = np.concatenate(( w,
                             np.zeros((V.size,)),
                             0*np.zeros((bH.size,)),
                             np.zeros((1,)) ))

    term2 = np.concatenate(( np.zeros((w.size,)),
                             np.zeros((V.flatten().size,)), #0*V.flatten(),
                             0*np.zeros((bH.size,)),
                             np.zeros((1,)) ))

    term3 = np.concatenate(( 1/nu * np.mean(deriv[:,np.newaxis] * (-hiddenScore(X, V, bH, g)), axis = 0),
                             1/nu * np.mean((deriv[:,np.newaxis] * (dG(X.dot(V) + bH) * -w)).reshape((N, 1, K)) * X.reshape((N, D, 1)), axis = 0).flatten(),
                             0*1/nu * np.mean((deriv[:,np.newaxis] * (dG(X.dot(V) + bH) * -w)).reshape((N, 1, K)) * np.ones((N, D, 1)), axis = 0).flatten(),
                             1/nu * np.array([ np.mean(deriv) ]) ))
    
    term4 = np.concatenate(( np.zeros((w.size,)),
                             np.zeros((V.size,)),
                             0*np.zeros((bH.size,)),
                             -1 * np.ones((1,)) ))
    
    return term1 + term2 + term3 + term4


class MyOCNN_FrozenB:

    def __init__(self, nu, b, K, g, dG):

        self.nu = nu
        self.b  = b
        self.K = K        
        self.g  = g
        self.dG = dG

    def fit(self, XTr, theta0):

        D = XTr.shape[1]
        K = self.K
        
        #theta0 = resEXP.x

        print('Gradient error: %s' % check_grad(ocfnn_obj, ocfnn_grad, theta0, XTr, self.nu, self.b, D, K, self.g, self.dG))
        resFNN = minimize(ocfnn_obj, theta0, 
                          jac     = ocfnn_grad, 
                          args    = (XTr, self.nu, self.b, D, K, self.g, self.dG),
                          method  = 'L-BFGS-B',                
                          options = {'gtol': 1e-16, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})
        print('Gradient error: %s' % check_grad(ocfnn_obj, ocfnn_grad, resFNN.x, XTr, self.nu, self.b, D, K, self.g, self.dG))

        self.w = resFNN.x[:K]
        self.V = resFNN.x[K:K+K*D].reshape((D,K))
        self.b = resFNN.x[K+K*D:K+K*D+K]
        self.r = resFNN.x[-1]
