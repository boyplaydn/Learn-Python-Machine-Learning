from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
from sklearn.svm import SVC

iris = datasets.load_iris()
iris_dataset = datasets.load_iris()
N = 50
X0 = iris.data[0:50, :2]  # we only take the Sepal two features.
X1 = iris.data[50:100, :2]
X = np.concatenate((X0.T, X1.T), axis = 1)

target = iris.target
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
print(y, 'y')
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((2*N, 1))) # all-one vector 
# build A, b, G, h 
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
print(b, 'b')
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)

x = np.array([5.2, 0.2])

mypredict = (w[0] * x[0] + w[1] * x[1] + b) / np.sqrt((w[0] * w[0] + w[1]*w[1]))

print('mypredict = ' , mypredict)


y1 = y.reshape((2*N,))
X1 = X.T # each sample is one row
clf = SVC(kernel = 'linear', C = 1e5) # just a big number 

clf.fit(X1, y1) 

w1 = clf.coef_
b1 = clf.intercept_

print('w1 = ', w1)
print('b1 = ', b1)

print(clf.predict([[5.2,  0.2]]), 'pre')