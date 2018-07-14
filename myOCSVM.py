import numpy as np
from scipy.optimize import minimize, check_grad

def relu(x):
    y = x
    y[y < 0] = 0
    return y

def dRelu(x):
    y = x
    y[x <= 0] = 0
    y[x > 0]  = np.ones((len(x[x > 0]),))
    return y

def svmScore(X, w):
    return X.dot(w)

def ocsvm_obj(theta, X, nu):
    
    w = theta[:-1]
    r = theta[-1]
    
    term1 = 0.5 * np.sum(w**2)
    term2 = 1/nu * np.mean(relu(r - svmScore(X, w)))
    term3 = -r
    
    return term1 + term2 + term3

def ocsvm_grad(theta, X, nu):
    
    w = theta[:-1]
    r = theta[-1]
    
    deriv = dRelu(r - svmScore(X, w))

    term1 = np.append(w, 0)
    term2 = np.append(1/nu * np.mean(deriv[:,np.newaxis] * (-X), axis = 0),
                      1/nu * np.mean(deriv))
    term3 = np.append(0*w, -1)

    grad = term1 + term2 + term3
    
    return grad


class MyOCSVM:

    def __init__(self, nu):

        self.nu = nu

    def fit(self, XTrTrans):

        K = XTrTrans.shape[1]
        theta0 = np.random.normal(0, 1, K + 1)
        print('Gradient error: %s' % check_grad(ocsvm_obj, ocsvm_grad, theta0, XTrTrans, self.nu))

        resEXP = minimize(ocsvm_obj, theta0, 
                          jac     = ocsvm_grad, 
                          args    = (XTrTrans, self.nu),
                          method  = 'L-BFGS-B',                
                          options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})

        self.w = resEXP.x[:-1]
        self.r = resEXP.x[-1]
