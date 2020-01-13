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
    
def mygrad1(x):
    return 3*x*x - 6*x

def mycost1(x):
    return (x**3 - 3*x**2)


def mySGD21(x0, eta = 0.1):
    x = [x0]
    N = 10
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            x_new = x[-1] - eta*mygrad1(x[-1])
            if abs(mygrad1(x_new)) < 0.0001:
                break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)

def plot_fn(fn, xmin = -5, xmax = 5, xaxis = True, opts = 'b-'):
    x = np.linspace(xmin, xmax, 1000)
    y = fn(x)
    ymin = np.min(y) - .5
    ymax = np.max(y) + .5
    plt.axis([xmin, xmax, ymin, ymax])
    if xaxis:
        x0 = np.linspace(xmin, xmax, 2)
        plt.plot([xmin, xmax], [0, 0], 'k')
    plt.plot(x, y, opts)
plot_fn(mycost1, -10, 10)
plt.show()

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


(x, it) = mySGD21(4)
viz_alg_1d(x, mycost1)
