import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


Xtrain = np.genfromtxt('data/Xtrain.csv', delimiter=',')
Ytrain = np.genfromtxt('data/ytrain.csv', delimiter=',')
Xtest = np.genfromtxt('data/Xtest.csv', delimiter=',')
Ytest = np.genfromtxt('data/ytest.csv', delimiter=',')
Q = np.genfromtxt('data/Q.csv', delimiter=',')
a = b = c = e = f = 1.0
d = len(Xtrain[0])

# Categories
N = np.zeros(2).astype(int)
N[0] = len(Ytrain[Ytrain==0])
N[1] = len(Xtrain)-N[0]
catIndex = list([np.argwhere(Ytrain==0).reshape(-1)])
catIndex.append(list(np.argwhere(Ytrain==1).reshape(-1)))

#Train
pi = np.zeros(2)
pi[0] = (f+N[0])/(sum(N)+e+f)
pi[1] = (e+N[1])/(sum(N)+e+f)

mu_0 = np.zeros((2,d))
lambda_0 = np.zeros((2,1))
alpha_0 = np.zeros((2,1))
beta_0 = np.zeros((2,d))

for i in range(2):
    x_mean = np.mean(Xtrain[catIndex[i][:],:], axis=0)
    mu_0[i,:] = (N[i]*x_mean)/(N[i]+1/a)
    lambda_0[i] = N[i]+1/a
    alpha_0[i,0] = b + N[i]/2.0
    beta_0[i,:] = c + 0.5*(N[i]*np.var(Xtrain[catIndex[i][:],:], axis=0)+np.square(x_mean)*(N[i]/a)/(N[i]+1/a))
    
# Predict:
likelihood = -np.ones((len(Ytest),2))
for i in range(2):
    likelihood[:,i] = np.prod(np.power((1 + lambda_0[i]*0.5*np.square(Xtest-mu_0[i])/((1+lambda_0[i])*beta_0[i])),-1*(alpha_0[i]+0.5)), axis=1)*pi[i]
label = np.argmax(likelihood, axis=1)

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
    prob = likelihood[errorIdx[i],label[errorIdx[i]]]/np.sum(likelihood[errorIdx[i],:])
    fig = plt.figure()            
    fig.suptitle('Misclassified, Bayes classification, true class: %s predicted: %s, Prob: %s'%(Ytest[errorIdx[i]], label[errorIdx[i]], prob), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
    
# Getting ambiguous indexes
errorIdx = np.argsort(np.abs((likelihood[:,0]-likelihood[:,1])/(likelihood[:,0]+likelihood[:,1])))[:3]

for i in range(3):
    pic = -np.dot(Q,Xtest[errorIdx[i],:].T).reshape(28,28)
    prob = likelihood[errorIdx[i],label[errorIdx[i]]]/np.sum(likelihood[errorIdx[i],:])
    fig = plt.figure()            
    fig.suptitle('Ambiguous, Bayes classification, true class: %s predicted: %s, Prob: %s'%(Ytest[errorIdx[i]], label[errorIdx[i]], prob), fontsize=18)
    plt.imshow(pic, cm.Greys_r)
    fig.show()
    


  


















