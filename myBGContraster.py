import numpy as np
from scipy.optimize import minimize, check_grad

def properObj(theta, XPos, XNeg, nu, gam, mu):

    w = theta[:-1][:,np.newaxis]
    r = theta[-1]

    term1 = np.maximum(r - XPos.dot(w), 0)
    term2 = mu * 0.5 * (XNeg.dot(w))**2
    term3 = -nu * r

    return np.mean(term1) + np.mean(term2) + term3 + (gam/2) * np.sum(w**2)


def properGrad(theta, XPos, XNeg, nu, gam, mu):

    w = theta[:-1][:,np.newaxis]
    r = theta[-1]

    term1  = r - XPos.dot(w)
    gradH  = (term1 > 0).astype(int) + 0.0 * (term1 == 0).astype(int)
    
    gradW1 = gradH * (-XPos)
    gradR1 = gradH * 1
    
    gradW2 = mu * (XNeg.dot(w)) * XNeg
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


class MyBGContraster:

    def __init__(self, nu):

        self.nu = nu

    def fit(self, XTrTrans, XBG):

        K = XTrTrans.shape[1]
        theta0 = np.random.normal(0, 1, K + 1)

        res = minimize(lambda w : properObj(w, XTrTrans, XBG, self.nu, 1e-2, 1), 
                       theta0,
                       jac = lambda w : properGrad(w, XTrTrans, XBG, self.nu, 1e-2, 1),
                       tol = 1e-16)

        self.w = res.x[:-1]
        self.r = res.x[-1]
