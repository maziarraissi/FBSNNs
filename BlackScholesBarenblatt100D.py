"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf
from FBSNNs import FBSNN
import matplotlib.pyplot as plt
from plotting import newfig, savefig

class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
               
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return 0.05*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # M x 1
    
    def g_tf(self, X): # M x D
        return tf.reduce_sum(X**2, 1, keepdims = True) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
        
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return 0.4*tf.matrix_diag(X) # M x D x D
    
    ###########################################################################

if __name__ == "__main__":
    
    M = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    
    layers = [D+1] + 4*[256] + [1]

    Xi = np.array([1.0,0.5]*int(D/2))[None,:]
    T = 1.0
         
    # Training
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers)
    
    model.train(N_Iter = 2*10**4, learning_rate=1e-3)
    model.train(N_Iter = 3*10**4, learning_rate=1e-4)
    model.train(N_Iter = 3*10**4, learning_rate=1e-5)
    model.train(N_Iter = 2*10**4, learning_rate=1e-6)
    
    ##### PLOT RESULTS
    
    t_test, W_test = model.fetch_minibatch()
    
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
        
    def u_exact(t, X): # (N+1) x 1, (N+1) x D
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max**2)*(T - t))*np.sum(X**2, 1, keepdims = True) # (N+1) x 1
        
    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M,:,:],[-1,1]), np.reshape(X_pred[0:M,:,:],[-1,D])),[M,-1,1])
    
    samples = 5
    
    plt.figure()
    plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1,:,0].T,Y_test[0:1,:,0].T,'r--',label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1,-1,0],Y_test[0:1,-1,0],'ko',label='$Y_T = u(T,X_T)$')
    
    plt.plot(t_test[1:samples,:,0].T,Y_pred[1:samples,:,0].T,'b')
    plt.plot(t_test[1:samples,:,0].T,Y_test[1:samples,:,0].T,'r--')
    plt.plot(t_test[1:samples,-1,0],Y_test[1:samples,-1,0],'ko')

    plt.plot([0],Y_test[0,0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    # savefig('./figures/BSB_Apr18_50', crop = False)
    
    
    errors = np.sqrt((Y_test-Y_pred)**2/Y_test**2)
    mean_errors = np.mean(errors,0)
    std_errors = np.std(errors,0)
    
    plt.figure()
    plt.plot(t_test[0,:,0],mean_errors,'b',label='mean')
    plt.plot(t_test[0,:,0],mean_errors+2*std_errors,'r--',label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    # savefig('./figures/BSB_Apr18_50_errors', crop = False)