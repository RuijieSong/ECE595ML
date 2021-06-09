import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import scipy.stats as stats

#exercise 3b
train_class0 = np.matrix(np.loadtxt('homework4_class0.txt'))
train_class1 = np.matrix(np.loadtxt('homework4_class1.txt'))
R0,C0 = train_class0.shape
R1,C1 = train_class1.shape

X = np.vstack((train_class1,train_class0))
X = np.hstack((X,np.ones(((R1+R0),1))))
y1 = np.ones((R1,1))
y0 = np.zeros((R0,1))
y = np.vstack((y1,y0))

_lambda = 0.0001
N = R1 + R0
theta = cvx.Variable((3,1))
L = -cvx.sum(cvx.multiply(y,X@theta)) \
    + cvx.sum(cvx.log_sum_exp(cvx.hstack([np.zeros((N,1)),X@theta]),axis=1))
reg = cvx.sum_squares(theta)
prob = cvx.Problem(cvx.Minimize(L/N + _lambda*reg))
prob.solve()
w = theta.value
print(w)

#exercise 3c
# w = w.flatten()
# b = -(w[2]*X[:,2]+w[0]*X[:,0])/w[1]
# plt.plot(X[:,0],b,linewidth='3',color='b')
# plt.plot(train_class0[:,0],train_class0[:,1],'o',markersize=10,color='g')
# plt.plot(train_class1[:,0],train_class1[:,1],'o',markersize=10,color='r')
# plt.axis([-6,10,-7,10])
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()

#exercise 3d

mu0 = np.matrix(np.mean(train_class0.T,axis=1)).T
mu1 = np.matrix(np.mean(train_class1.T,axis=1)).T
sigma0 = np.cov(train_class0.T)
sigma1 = np.cov(train_class1.T)
K0 = len(train_class0.T)
K1 = len(train_class1.T)
pi0 = K0/(K0+K1)
pi1 = K1/(K0+K1)

x = np.linspace(-5,10,100)
y = np.linspace(-5,10,100)
boundary = np.zeros((100,100))

for i in range(100):
    for j in range(100):
        block = np.matrix([x[i],y[j]]).T
        a = -0.5*(block-mu0).T*np.linalg.inv(sigma0)*(block-mu0) \
            + np.log(pi0) - 0.5*np.log(np.linalg.det(sigma0))
        b = -0.5*(block-mu1).T*np.linalg.inv(sigma1)*(block-mu1) \
            + np.log(pi1) - 0.5*np.log(np.linalg.det(sigma1))
        if (a[0][0] > b[0][0]).all():
            boundary[i,j]=1

plt.scatter(train_class0[:,0].tolist(),train_class0[:,1].tolist(),marker='o',s=50)
plt.scatter(train_class1[:,0].tolist(),train_class1[:,1].tolist(),marker='x',s=50)
plt.contour(x,y,boundary>0,linewidth = 1,c='c')
plt.show() 
  


