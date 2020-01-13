from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
import matplotlib.pyplot as plt

def grad(x):
    return 2*x+ 10*np.cos(x)

def cost(x):
    return x**2 + 10*np.sin(x)

def mygrad(x):
    return (x*x - 2*x) / (x-1)*(x-1)

def mycost(x):
    return (x**2)/(x-1)

def mygrad1(x):
    return 3*x*x - 6*x

def mycost1(x):
    return (x**3 - 3*x**2)

def mygrad2(x):
    return (2*x + 10*np.cos(x))

def mycost2(x):
    return (x*x + 10*np.sin(x))


def Momentum(x0, eta = 0.1, gamma = 0.9):
    x = [x0]
    v = [0]
    
    for it in range(100):
        g = grad(x[-1])
        v_new = gamma*v[-1] + eta*g
        x_new = x[-1] - v_new
        if abs(grad(x_new)) < 1e-3:
            break
        v.append(v_new)
        x.append(x_new)
        #
    return (np.asarray(x), v, it)

def viz_alg_1d(x, cost, filename = 'momentum1d2.gif'):
#     x = x.asarray()
    it = len(x)
    y = cost(x)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    
    xmin, xmax = -4, 6
    ymin, ymax = -12, 25
    
    x0 = np.linspace(xmin-1, xmax+1, 1000)
    y0 = cost(x0)
       
    fig, ax = plt.subplots(figsize=(4, 4))  
    
    def update(i):
        ani = plt.cla()
        plt.axis([xmin, xmax, ymin, ymax])
        plt.plot(x0, y0)
        #ani = plt.title('$f(x) = x^2 + 10\sin(x); x_0 = 5; \eta = 0.1; \gamma = 0.9$')
        if i == 0:
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        else:
            ani = plt.plot(x[i-1], y[i-1], 'ok', markersize = 7)
            ani = plt.plot(x[i-1:i+1], y[i-1:i+1], 'k-')
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        label = 'GD with Momemtum: iter %d/%d' %(i, it)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    anim.save(filename, dpi = 100, writer = 'imagemagick')
    plt.show()

# x = np.asarray(x)
(x, v, it) = Momentum(6, 0.09, 0.7)
# (x) = GD_newton1(4)
# (x, it) = myGD21(0,0.09)
#(x1, it1) = myGD21(4, 0.09)
# print(x[-1], it)
print(x[-1], it)
viz_alg_1d(x, cost)