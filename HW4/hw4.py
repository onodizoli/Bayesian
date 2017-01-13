import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, psi, multigammaln
from scipy.stats import wishart, dirichlet, multivariate_normal, rv_discrete


X = np.genfromtxt('data/data.csv', delimiter=',')
np.random.seed(19900919)


# Q1
N, d = np.shape(X)
K = [2,4,8,10]
T = 100

# Loop over number of clusters
likelihoods = []
for k in K:
    pi = np.random.rand(k)
    # Normalize to be a prob dist
    pi /= sum(pi)
    mu = np.random.rand(k,d)
    lamb = np.asarray([np.eye(d)*np.random.rand(1) for i in range(k)])
    #lamb = np.random.rand(k,d,d)
    fi = np.zeros((k,N))
    loglike = []
    log = np.zeros((k,N))
    # Loop over iteration
    for t in range(T):
        for j in range(k):
            fi[j,:] = pi[j] * np.diag((np.linalg.det(lamb[j,:,:])**0.5)/(2*np.pi)*np.exp(-0.5*np.dot(X-mu[j,:],(np.dot(lamb[j,:,:],(X-mu[j,:]).T)))))
        fi = fi / np.sum(fi,axis = 0)
        n = np.sum(fi, axis = 1)
        for j in range(k):
            mu[j,:] = np.sum(X.T*fi[j,:],axis = 1)/n[j]
            lamb[j,:,:] = np.linalg.inv(np.dot((X-mu[j,:]).T*fi[j,:],X-mu[j,:])/n[j])
        pi = n/N
        # Calculate loglike
        for j in range(k):
            log[j,:] = pi[j] * np.diag((np.linalg.det(lamb[j,:,:])**0.5)/(2*np.pi)*np.exp(-0.5*np.dot(X-mu[j,:],(np.dot(lamb[j,:,:],(X-mu[j,:]).T)))))
        loglike.append(np.sum(np.log(np.sum(log,axis=0))))
    likelihoods.append(loglike)
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], c=np.argmax(fi, axis=0), marker='+')
    fig.suptitle('EM-GMM with K = %s'%(k))
fig = plt.figure()
for i in range(len(K)):
    plt.plot(range(T),likelihoods[i])
plt.legend(K, loc=4)
fig.suptitle('EM-GMM log likelihood with different number of clusers')

   
#Q2 
np.random.seed(199009191)
def expLnLamb(a, d, B):
    return d*np.log(2)-np.log(np.linalg.det(B))+np.sum([psi(a/2.0+(1-i)/2.0) for i in range(1,d+1)], axis=0)
    
def multinomEntropy(c, phi):
    res = -np.sum([phi[i]*np.log(phi[i]) for i in range(len(phi))])+phi[c]
    return res
    
    
N, d = np.shape(X)
K = [2,4,10,25]
T = 100
alpha_0 = 1.0
c_0 = 10.0
a_0 = float(d)
A = np.cov(X.T)
B_0 = d/10.0*A

#initialize params
for k in K:
    alpha = np.ones(k)*alpha_0/float(k)
    m = np.random.multivariate_normal([0,0], np.eye(d)*c_0, k)
    #m = X[np.random.randint(0,N,k),:]
    sigma = np.asarray([A for i in range(k)])
    a = np.ones(k)*a_0
    B = np.asarray([B_0 for i in range(k)])
    n = np.zeros(k)
    phi = np.zeros((k,N))
    c = np.zeros((N))
    #B = np.asarray([wishart.rvs(a_0,np.linalg.inv(B_0)) for i in range(k)])
    objective = []
    for t in range(T):
        
        #P(mu) check sign
        o_2 = -np.sum([1/2.0/c_0*((np.dot(m[i,:],m[i,:].T)) + np.trace(sigma[i,:,:])) for i in range(k)])
        #P(lamb)
        o_3 = np.sum((a_0-d-1)/2.0*expLnLamb(a, d, B)-0.5*np.asarray([np.trace(np.dot(B_0,a[i]*np.linalg.inv(B[i,:,:]))) for i in range(k)])-a_0*d/2*np.log(2)+a_0/2*np.log(np.linalg.det(B_0))-multigammaln(a_0,2))
        #P(pi)
        o_4 = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((psi(alpha) - psi(np.sum(alpha)))*(alpha-1)) #working
        #q(mu_j) entropy    
        o_5 = np.sum([multivariate_normal.entropy(mean=m[i,:],cov=sigma[i,:,:]) for i in range(k)])
        #q(lamb_j) entropy
        o_6 = np.sum([wishart.entropy(a[i],np.linalg.inv(B[i,:,:])) for i in range(k)])
        #q(pi)
        o_7 = dirichlet.entropy(alpha)
        #q(c_i)
        o_8 = np.sum([multinomEntropy(c[i],phi[:,i]) for i in range(N)])
        # (a)
        t_1 = np.sum([psi((1-i+a)/2) for i in range(1,d+1)], axis=0)-np.log(np.linalg.det(B))
        t_2 = np.zeros((k,N))
        t_3 = np.zeros(k)
        for i in range(k):
            t_2[i,:] = np.diag(np.dot(X-m[i,:],np.dot(a[i]*np.linalg.inv(B[i,:,:]),(X-m[i,:]).T)))
            t_3[i] = np.trace(a[i]*np.dot(np.linalg.inv(B[i,:,:]),sigma[i,:,:]))        
        t_4 = psi(a) - psi(sum(a))
        phi= np.exp(0.5*t_1-0.5*t_2.T-0.5*t_3+t_4)
        phi = phi.T/np.sum(phi, axis=1)
        
        n = np.sum(phi, axis=1)
        
        alpha = alpha_0 + n
        
        for i in range(k):
            sigma[i,:,:] = np.linalg.inv(np.eye(d)/c_0 + n[i]*a[i]*np.linalg.inv(B[i,:,:]))
            m[i,:] = np.dot(sigma[i,:,:],np.dot(a[i]*np.linalg.inv(B[i,:,:]),np.sum((X.T*phi[i,:]),axis=1)))
         
        a = a_0 + n
        for i in range(k):
            # double check phi part
            B[i,:,:] = B_0 + np.dot((X-m[i,:]).T*phi[i,:],X-m[i,:]) + sigma[i,:,:]*np.sum(phi[i,:])
        c = np.argmax(phi, axis=0)    
        #objective
        #P(x)
        o_1 = np.sum([(0.5*t_1-0.5*t_2.T-0.5*t_3+t_4)[i,c[i]] for i in range(N)])
        objective.append(o_1+o_2+o_3+o_4+o_5+o_6+o_7 + o_8)
    fig = plt.figure()
    plt.plot(range(1,T),objective[1:])         
    fig.suptitle('Objective func of Variational Inference for GMM with K = %s'%(k)) 
    plt.xlabel('Iteration')       
            

    labels = [[i,list(c).count(i)] for i in set(c)]
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], c=c, marker='+')
    fig.suptitle('Variational Inference for GMM with K = %s'%(k))
        


#Q3
np.random.seed(199009)
def sampleTheta(data, m, c, a, B):
    s = np.shape(data)[0]
    m_prime = c/(s+c)*m+np.sum(data,axis=0)/(s+c)
    c_prime = s+c
    a_prime = a+s
    avg = np.mean(data, axis=0)
    B_prime = B + np.dot((data-avg).T,(data-avg))+s/(a*s+1)*np.outer(avg-m,avg-m)
    #B_prime = B + np.dot((data-avg).T,(data-avg))+s*c/(s+c)*np.outer(avg-m,avg-m)
    lamb = wishart.rvs(a_prime,np.linalg.inv(B_prime))
    mu = multivariate_normal.rvs(m_prime, np.linalg.inv(c_prime*lamb))
    return [mu, np.linalg.inv(lamb)]

def xMarginal(x, d, m, c, a, B):
    expterm = np.exp(np.sum([gammaln((a+1)/2.0+(1-k)/2.0)-gammaln(a/2.0+(1-k)/2.0) for k in range(1,d+1)]))
    result = (np.power(c/(np.pi*(1+c)), 0.5*d)*np.power(np.linalg.det(B+c/(1.0+c)*np.outer(x-m,x-m)),(a+1)/-2.0)
                /np.power(np.linalg.det(B), a/-2.0)*expterm)
    return result
                

N, d = np.shape(X)
T = 500
m = np.mean(X, axis=0)
c = 1.0/10
a = float(d)
A = np.cov(X.T)
B = c*d*A
alpha = 1.0
ci = np.zeros(N)
theta = []
theta.append(sampleTheta(X, m, c, a, B))
theta_idx = np.asarray(0).reshape(1)
n = np.asarray([0, N]).reshape(1,2)
obsPerClust = []
numOfClust = []
for t in range(T):
    if t%(T/10)==0:
        print str((t/(T/10)*10)) + '% is done'
    for i in range(N):
        phi = []
        classes = []
        for j in range(len(n)):
            if ci[i] == n[j][0] and n[j][1] >= 2:
                classes.append(n[j][0])
                phi.append(multivariate_normal.pdf(X[i,:], mean=theta[j][0], cov=theta[j][1])*(n[j][1]-1)/(alpha+N-1))
            elif ci[i] != n[j][0]:
                classes.append(n[j][0])
                phi.append(multivariate_normal.pdf(X[i,:], mean=theta[j][0], cov=theta[j][1])*(n[j][1])/(alpha+N-1))
        phi.append(alpha/(alpha+N-1)*xMarginal(X[i,:], d, m, c, a, B))
        phi = np.asarray(phi)
        phi /= sum(phi)
        j_prime = max(n[:,0])+1
        classes.append(j_prime)
        classes = np.asarray(classes)
        sampler = rv_discrete(name='sampler', values=(classes, phi))
        c_gen = sampler.rvs()
        #c_gen = sampler(phi)
        if ci[i] != c_gen:
            n_idx = int(np.argwhere(n[:,0]==ci[i]))
            n[n_idx,1] -= 1
            if n[n_idx,1].reshape(-1) == 0:
                del(theta[int(np.argwhere(theta_idx==int(ci[i])))])
                theta_idx = np.delete(theta_idx,np.argwhere(theta_idx==int(ci[i])))
                n = np.delete(n, n_idx,0)
            if c_gen == j_prime:
                n = np.concatenate((n,np.asarray([j_prime,1]).reshape(1,2)))
                theta.append(sampleTheta(X[i,:], m, c, a, B))
                theta_idx = np.concatenate((theta_idx,np.asarray([j_prime]).reshape(1)))
                ci[i] = c_gen
            if c_gen != j_prime:
                new_idx = int(np.argwhere(n[:,0]==c_gen))
                n[new_idx,1] += 1
                ci[i] = c_gen

    theta = []
    theta_idx = np.asarray([])
    used_classes = list(set(n[:,0]))
    for j in range(len(used_classes)):
        n_idx = np.argwhere(n[:,0]==used_classes[j]).reshape(-1)
        databin = np.argwhere(ci==used_classes[j]).reshape(-1)               
        theta.append(sampleTheta(X[databin,:], m, c, a, B))
        theta_idx = np.concatenate((theta_idx,np.asarray([j]).reshape(1)))
        ci[databin] = j
        n[n_idx,0] = j

    clusters = np.sort([list(ci).count(i) for i in set(ci)])[::-1]
    numOfClust.append(len(clusters))
    if len(clusters)<6:
        clusters = np.concatenate((clusters,[0 for j in range(6-len(clusters))]))
    clusters = clusters[:6]
    obsPerClust.append(clusters)
    
obsPerClust = np.asarray(obsPerClust)
numOfClust = np.asarray(numOfClust)           
                
fig = plt.figure()
for j in range(6):
    plt.plot(range(T), obsPerClust[:,j])
fig.suptitle('Observations per cluster for top 6')
plt.xlabel('Iteration')
plt.ylabel('Number of datapoint in class')                   
                
fig = plt.figure()
plt.plot(range(T), numOfClust)                
fig.suptitle('Number of clusters containing data')
plt.xlabel('Iteration')
plt.ylabel('Number of classes')         
                
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c=ci, marker='+')
fig.suptitle('Gibbs sampling for GMM')                
