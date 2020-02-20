"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf
from FBSNNs import FBSNN
import matplotlib.pyplot as plt
from plotting import newfig, savefig

class AllenCahn(FBSNN):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return - Y + Y**3 # M x 1
    
    def g_tf(self, X):
        return 1.0/(2.0 + 0.4*tf.reduce_sum(X**2, 1, keepdims = True))

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return super().sigma_tf(t, X, Y) # M x D x D
    
    ###########################################################################


if __name__ == "__main__":
    
    M = 100  # number of trajectories (batch size)
    N = 15   # number of time snapshots
    D = 20   # number of dimensions
    
    layers   = [D+1] + 4*[256] + [1]

    T = 0.3
    Xi = np.zeros([1,D])
         
    # Training
    model = AllenCahn(Xi, T,
                      M, N, D,
                      layers)
        
    model.train(N_Iter = 2*10**4, learning_rate=1e-3)
    model.train(N_Iter = 3*10**4, learning_rate=1e-4)
    model.train(N_Iter = 3*10**4, learning_rate=1e-5)
    model.train(N_Iter = 2*10**4, learning_rate=1e-6)
    
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    
    samples = 5
    
    Y_test_terminal = 1.0/(2.0 + 0.4*np.sum(X_pred[:,-1,:]**2, 1, keepdims = True))
    
    plt.figure()
    plt.plot(t_test[0,:,0].T,Y_pred[0,:,0].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t_test[1:samples,:,0].T,Y_pred[1:samples,:,0].T,'b')
    plt.plot(t_test[0:samples,-1,0],Y_test_terminal[0:samples,0],'ks',label='$Y_T = u(T,X_T)$')
    plt.plot([0],[0.30879],'ko',label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('20-dimensional Allen-Cahn')
    plt.legend()
    
    # savefig('./figures/AC_Apr18_15', crop = False)