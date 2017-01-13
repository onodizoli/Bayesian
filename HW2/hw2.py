import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

Xtrain = np.genfromtxt('data/Xtrain.csv', delimiter=',')
Ytrain = np.genfromtxt('data/ytrain.csv', delimiter=',')
Xtest = np.genfromtxt('data/Xtest.csv', delimiter=',')
Ytest = np.genfromtxt('data/ytest.csv', delimiter=',')
Q = np.genfromtxt('data/Q.csv', delimiter=',')


# Parameters
sigma = 1.5
lambd =1.0 
T = 100

omega = np.zeros((len(Xtest[0,:]),1))
#omega = np.random.rand(len(Xtest[0,:]),1)
likelihood = []
wplot = []
plotrange = [1,5,10,25,50,100]

def trueLabel(x):
    if x ==0:
        return 4
    else:
        return 9


    
for t in range(T):
    X = np.dot(-Xtrain,omega)/sigma
    pdf = norm(0,1).pdf(X)
    cdf = -norm(0,1).cdf(X)
    denom = cdf
    denom[np.argwhere(Ytrain==1)] += 1
    E = np.dot(Xtrain,omega) + sigma*pdf/denom
    omega = np.dot(np.linalg.inv(np.eye(len(Xtrain[0,:]))*lambd + \
        np.dot(Xtrain.T,Xtrain)/(sigma**2)),np.sum(Xtrain*E/(sigma**2), axis=0)).reshape(15,1)
    likelihood.append(15/2.0*np.log(lambd/(2*np.pi))-lambd/2*np.dot(omega.T,omega).reshape(-1) + \
        np.sum(np.log(norm(0,1).cdf(np.dot(Xtrain,omega)/sigma))*Ytrain.reshape(len(Ytrain),1)) + \
            np.sum(np.log(1-norm(0,1).cdf(np.dot(Xtrain,omega)/sigma))*(1-Ytrain).reshape(len(Ytrain),1)))
    if t+1 in plotrange:
        wplot.append(omega)

fig = plt.figure()
fig.suptitle('ln P(y,w_t|X')    
plt.xlabel('Iteration')
plt.ylabel('likelihood')    
plt.ylim([-4500,-1900]) 
plt.plot(range(T),likelihood)
fig.show()
    
like = norm(0,1).cdf(np.dot(Xtest,omega)/sigma).reshape(-1)
label =  np.round(like)

confusion = np.zeros((2,2))
for i in range(2):
    confusion[i,i] = len(np.argwhere(Ytest[Ytest==i]==label[Ytest==i]))
    confusion[i,(i+1)%2] = len(np.argwhere(Ytest[Ytest==i]!=label[Ytest==i]))

correct = len(np.argwhere(Ytest==label))
precision = correct/float(len(Ytest))
print precision
print confusion


# Plots
index0 = np.argwhere(Ytest[Ytest==0]!=label[Ytest==0])[:2]
index1 = len(Ytest[Ytest==0])+np.argwhere(Ytest[Ytest==1]!=label[Ytest==1])[:1].reshape(-1)
errorIdx = np.append(index0,index1)
for i in range(3):
    pic = -np.dot(Q,Xtest[errorIdx[i],:].T).reshape(28,28)
    prob = like[errorIdx[i]]
    fig = plt.figure()            
    fig.suptitle('Misclassified, true class: %s predicted: %s, Prob: %s'%(trueLabel(Ytest[errorIdx[i]]), trueLabel(label[errorIdx[i]]), prob), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
    
# Getting ambiguous indexes
errorIdx = np.argsort(np.abs(like-0.5))[:3]

for i in range(3):
    pic = -np.dot(Q,Xtest[errorIdx[i],:].T).reshape(28,28)
    prob = like[errorIdx[i]]
    fig = plt.figure()            
    fig.suptitle('Ambiguous, true class: %s predicted: %s, Prob: %s'%(trueLabel(Ytest[errorIdx[i]]), trueLabel(label[errorIdx[i]]), prob), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
        
for i in range(len(wplot)):
    pic = -np.dot(Q,wplot[i]).reshape(28,28)
    fig = plt.figure()            
    fig.suptitle('Omega_t treated as image,  t = %s '%(plotrange[i]), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()    
