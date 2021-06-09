import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
from numpy.matlib import repmat

class0 = np.loadtxt('homework4_class0.txt')
class1 = np.loadtxt('homework4_class1.txt')

N0 = class0.shape[0]
N1 = class1.shape[0]
N  = N0 + N1
y0 = np.zeros((N0,1))
y1 = np.ones((N1,1))
x = np.vstack((class0,class1))
y = np.vstack((y0,y1))
X = np.hstack((x, np.ones((N,1))))

#exercise 4a
h = 1
K  = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    K[i,j]  = np.exp(-np.sum((X[i,:]-X[j,:])**2)/h)

print("The  K[47:52,47:52] is:")
np.set_printoptions(precision=5, suppress=True)
print(K[47:52, 47:52])


#exercise 4c
lambd = 0.001
alpha = cvx.Variable((N,1))
loss = -cvx.sum(cvx.multiply(y,  K @ alpha)) + cvx.sum(cvx.log_sum_exp(cvx.hstack([np.zeros((N,1)), K@alpha]), axis = 1))
reg = cvx.sum(cvx.quad_form(alpha, K))
prob = cvx.Problem(cvx.Minimize(loss/N + lambd*reg))
prob.solve()
w = alpha.value
print("The first two elements of the regression coefficients is:")
print(w[0:2, 0])

# exercise 4d
xset = np.linspace(-5,10,100)
yset = np.linspace(-5,10,100)
output = np.zeros((100,100))
for i in range(100):
  for j in range(100):
    data = repmat( np.array([xset[j], yset[i], 1]).reshape((1,3)), N, 1)
    phi  = np.exp( -np.sum( (X-data)**2, axis=1 )/h )
    output[i,j] = np.dot(phi.T, w)

# Display
plt.scatter(class0[:,0], class0[:,1],marker='o',s=20)
plt.scatter(class1[:,0], class1[:,1],marker='+',s=60)
plt.contour(xset, yset, output>0.5, linewidths=2, colors='k')
plt.show()
