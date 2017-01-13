import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, psi




for k in range(1,4):
    Xset = np.genfromtxt('data/X_set' + str(k) + '.csv', delimiter=',')
    Yset = np.genfromtxt('data/y_set' + str(k) + '.csv', delimiter=',')
    Zset = np.genfromtxt('data/z_set' + str(k) + '.csv', delimiter=',')
    

    # Parameters
    a_0 = np.power(10.0,-16)
    b_0 = np.power(10.0,-16)
    e_0 = 1.0
    f_0 = 1.0
    T = 500    
    N = Xset.shape[0]
    d = Xset.shape[1]
    # Variational Inference
    e = e_0
    f = f_0
    a = a_0
    b = np.ones(Xset.shape[1])*b_0
    L = []
    for t in range(T):
        sigma = np.linalg.inv(np.diag(a/b) + e/f*np.dot(Xset.T,Xset))
        mu = np.dot(sigma,e/f*np.sum(Yset.reshape(N,1)*Xset,axis=0))
        e = e_0 + N/2.0
        f = f_0 + 1.0/2*np.sum(np.power(Yset-np.dot(Xset,mu),2)+np.diag(np.dot(np.dot(Xset,sigma),Xset.T)))
        a = a_0 + 1/2.0
        b = b_0 + 1/2.0*(sigma.diagonal()+mu**2)
        L1 = N/2.0*(psi(e)-np.log(f))-e/2.0/f*np.sum(((Yset-np.dot(Xset,mu))**2)+np.diag(np.dot(np.dot(Xset,sigma),Xset.T)))
        L2 = 1/2.0*np.sum(psi(a)-np.log(b))-1/2.0*np.sum((sigma.diagonal()+mu**2)*a/b)
        L3 = (a_0-1)*np.sum(psi(a)-np.log(b))-b_0*np.sum(a/b)
        L4 = (e_0-1)*(psi(e)-np.log(f))-f_0*e/f
        L5 = 1/2.0*np.prod(np.linalg.slogdet(sigma))
        L6 = (e-np.log(f)+gammaln(e)+(1-e)*psi(e))
        L7 = np.sum(a-np.log(b)+gammaln(a)+(1-a)*psi(a))
        #L8 = e_0*np.log(f_0)-gammaln(e_0)+d*(a_0*np.log(b_0)-gammaln(a_0)) + d/2.0*np.log(2*np.pi*np.e)-N/2*(np.log(2*np.pi)) #constant term
        L.append(L1+L2+L3+L4+L5+L6+L7)#+L8)

        
    fig = plt.figure()
    fig.suptitle('Variational objective function, N = %s'%(N))
    plt.xlabel('Iteration')
    #plt.ylabel('likelihood')    
    plt.ylim([min(L)*0.9,max(L)*1.1])
    plt.plot(range(T),L)
    
    
    fig = plt.figure()
    fig.suptitle('1/E[alpha_k], N = %s'%(N))  
    plt.xlabel('k')
    plt.ylabel('1/E[alpha_k]')    
    #plt.ylim([-4500,-1900]) 
    plt.stem(range(len(b)),b/a)
    
    print(' 1/E[lambda] =  ' + str(f/e))
    
    yhat = np.dot(Xset,mu)
    fig = plt.figure()
    fig.suptitle('N = %s'%(N))
    plt.plot(Zset,yhat, c='r')
    plt.scatter(Zset, Yset, marker='+')
    plt.plot(Zset, 10*np.sinc(Zset), c='g' )
    plt.plot()
    plt.legend(('y_hat','10*sinc(z)','y'))
    plt.xlabel('z_i')
    fig.show()

