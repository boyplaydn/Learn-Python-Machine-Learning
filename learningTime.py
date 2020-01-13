from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
import matplotlib.pyplot as plt



X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

#ghep so 1 vao moi phan tu cua X de phuong trinh co dang sau khi nhan 2 matrix la ax+b
# X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

X=X.T 
y = y.T

X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)


def sigmoid(s):
    return 1/(1 + np.exp(-s))

def mygrad(theta):
    return 1/(1 + np.exp(-np.dot(X,theta.T)))

def myGD(X, y, theta_init, eta = 0.05):
    theta_old = theta_init
    theta_epoch = theta_init
    N = X.shape[0]

    for it in range(10000):
        for i in range(N):
            xi = X[i, :]
            yi = y[i]
            hi = 1.0/(1 + np.exp(-np.dot(xi,theta_old.T)))
            gi = (yi - hi)*xi
            theta_new = theta_old + eta*gi
            theta_old = theta_new


        if np.linalg.norm(theta_epoch - theta_old) < 1e-3:
            break

        theta_epoch = theta_old

    
    return theta_epoch, it

def mySGD(X, y, theta_init, eta = 0.05):
    theta_old = theta_init
    theta_epoch = theta_init
    N = X.shape[0]
    for it in range(10000):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :]
            yi = y[i]
            hi = 1.0/(1 + np.exp(-np.dot(xi,theta_old.T)))
            gi = (yi - hi)*xi
            theta_new = theta_old + eta*gi
            theta_old = theta_new
        if np.linalg.norm(theta_epoch - theta_old) < 1e-3:
            break
        theta_epoch = theta_old

    
    return theta_epoch, it





theta_init = np.array([1.5, 0.5])

theta, it = mySGD(X, y, theta_init)
print(theta, it)

def predict() :
    hours = float(input("Enter hours"))
    z = theta[0]*hours + theta[1]
    total = 1/(1 + np.exp(-z))
    print(total)
    return total

predict()