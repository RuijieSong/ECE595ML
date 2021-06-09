# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:21:35 2021

@author: 11327
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from numpy.matlib import repmat

# Ex 3
# b
# read data from txt file
xclass0 = np.matrix(np.loadtxt('./data/quiz4_class0.txt'))
xclass1 = np.matrix(np.loadtxt('./data/quiz4_class1.txt'))

# create x
x = np.concatenate((xclass0,xclass1),axis=0)
[rowx,colx] = np.shape(x)

# create y
[rowx0,colx0] = np.shape(xclass0)
[rowx1,colx1] = np.shape(xclass1)
y0 = np.zeros((rowx0,1))
y1 = np.ones((rowx1,1))
y = np.concatenate((y0,y1),axis=0)

# CVX
lambd = 0.01
N = rowx0 + rowx1
x = np.hstack((x, np.ones((N,1))))
theta       = cvx.Variable((3,1))
loss        = - cvx.sum(cvx.multiply(y, x @ theta)) \
              + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((N,1)), x @ theta]), axis=1 ) )
reg         = cvx.sum_squares(theta)
prob        = cvx.Problem(cvx.Minimize(loss/N + lambd*reg))
prob.solve()
w = theta.value

# c
# calculate the boundary
xb = np.linspace(-4,8,100)
yb = (-w[0]*xb-w[2])/w[1]

# do the plot
'''
plt.figure()
plt.scatter(xclass0[:,0].tolist(),xclass0[:,1].tolist())
plt.scatter(xclass1[:,0].tolist(),xclass1[:,1].tolist(), c='g')
'''
"""
plt.plot(xb,yb,c='b')
plt.show()
"""
# d
# create testing sites
n = 100
testing = np.linspace(-5,10,n)
# y = np.linspace(-5,10,n)
xv,yv = np.meshgrid(testing,testing)

boundary = np.zeros((n,n))

# find parameters
miu0 = np.zeros(colx0)
for i in range(colx0):
    miu0[i] = np.mean(xclass0[i])

miu1 = np.zeros(colx1)
for i in range(colx1):
    miu1[i] = np.mean(xclass1[i])
    
Sigma0 = np.cov(xclass0.T)
Sigma1 = np.cov(xclass1.T)

d = rowx0
abs_Sigma1 = np.linalg.det(Sigma1)
abs_Sigma0 = np.linalg.det(Sigma0)
inv_Sigma1 = np.linalg.inv(Sigma1)
inv_Sigma0 = np.linalg.inv(Sigma0)
const = np.power((2*np.pi),d)

K0 = len(xclass0.T)
K1 = len(xclass1.T)
pi0 = K0/(K0+K1)
pi1 = K1/(K0+K1)

# do Bayesian Decision
for i in range(100):
    for j in range(100):
        block = np.matrix([testing[i],testing[j]]).T
        # block = np.matrix((x[i,0],x[i,1])).T
        a = -0.5*(block-miu0).T*np.linalg.inv(Sigma0)*(block-miu0) \
            + np.log(pi0) - 0.5*np.log(np.linalg.det(Sigma0))
        b = -0.5*(block-miu1).T*np.linalg.inv(Sigma1)*(block-miu1) \
            + np.log(pi1) - 0.5*np.log(np.linalg.det(Sigma1))
        if (a[0][0] > b[0][0]).all():
            boundary[i,j]=1
'''      
plt.contour(testing,testing,boundary)
plt.show()
'''
# Ex 4
# a
'''
m,n = 100,100
K = np.zeros((m,n))

h = 1
x = x[:,0:2]
for i in range(m):
    for j in range(n):
        K[i,j] = np.exp(-np.power(np.linalg.norm(x[i,:]-x[j,:],ord=1),2)/h)
'''        
# print(K[47:52,47:52])

# x = x[:,0:2]
h = 1
K  = np.zeros((N,N))
# X = x
for i in range(N):
    for j in range(N):
        K[i,j]  = np.exp(-np.linalg.norm((x[i,:]-x[j,:]))**2 /h)

print("The  K[47:52,47:52] is:")
np.set_printoptions(precision=5, suppress=True)
print(K[47:52, 47:52])
    
# c

lambd = 0.001
alpha     = cvx.Variable((N,1))
loss        = - cvx.sum(cvx.multiply(y, K @ alpha)) \
              + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((N,1)), K @ alpha]), axis=1 ) )
reg         = cvx.sum(cvx.quad_form(alpha, K))
prob        = cvx.Problem(cvx.Minimize(loss/N + lambd*reg))
prob.solve()
ALPHA = alpha.value

'''
lambd = 0.001
alpha = cvx.Variable((N,1))
loss = -cvx.sum(cvx.multiply(y,  K @ alpha)) + cvx.sum(cvx.log_sum_exp(cvx.hstack([np.zeros((N,1)), K@alpha]), axis = 1))
reg = cvx.sum(cvx.quad_form(alpha, K))
prob = cvx.Problem(cvx.Minimize(loss/N + lambd*reg))
prob.solve()
w = alpha.value
print("The first two elements of the regression coefficients is:")
print(w[0:2, 0])
''' 

# d

xset = np.linspace(-5,10,100)
yset = np.linspace(-5,10,100)
output = np.zeros((100,100))
for i in range(100):
  for j in range(100):
    data = repmat( np.array([xset[j], yset[i], 1]).reshape((1,3)), N, 1)
    phi  = np.exp( -np.sum( (np.array(x)-data)**2, axis=1)/h )
    output[i,j] = np.dot(phi.T, ALPHA)

# Display
plt.figure()
plt.scatter(xclass0[:,0].tolist(), xclass0[:,1].tolist())
plt.scatter(xclass1[:,0].tolist(), xclass1[:,1].tolist(), c='g')
plt.contour(xset, yset, output>0.5, linewidths=2, colors='k')
plt.show()  






























