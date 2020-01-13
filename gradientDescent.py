from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
import matplotlib.pyplot as plt


eta = .05
x0 = 5
c_sin = 5
title = '$f(x) = x^2 + %dsin(x)$; ' %c_sin
title += '$x_0 =  %.2f$; ' %x0
title += r'$\eta = %.2f$ ' % eta 
file_name = 'gd_14.gif'
def grad(x):
    return 2*x+ 10*np.cos(x)

def grad2(x):
    return 2 - 10*np.sin(x)

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

#phuong phap gradient co dien
def myGD1(x0, eta = 0.1):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)

# (x, it) = myGD1(5, 0.1)
# print(x[-1], it)

def myGD2(x0, eta = 0.1):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*mygrad(x[-1])
        if abs(mygrad(x_new)) < 0.00001:
            break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)

# (x, it) = myGD2(0.1,0.09)
# (x1, it1) = myGD2(4, 0.09)
# print(x[-1], it)
# print(x1[-1], it1)

def myGD21(x0, eta = 0.1):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta*mygrad1(x[-1])
        if abs(mygrad1(x_new)) < 0.0001:
            break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)

(x, it) = myGD21(3,0.09)
(x1, it1) = myGD21(4, 0.09)
print(x[-1], it)
print(x1[-1], it1)

#phuong phap Gradient
def GD_newton(x0):
    x = [x0]
    for it in range(100):
        if abs(cost(x[-1])) < 1e-6 or abs(grad(x[-1])) < 1e-6:
            break
        x_new = x[-1] - 3*cost(x[-1])/grad(x[-1])
        #print(x_new, cost(x[-1]), grad(x[-1]))
        x.append(x_new)
    return (x, it)

def GD_newton1(x0):
    x = [x0]
    for it in range(100):
        if abs(mycost1(x[-1])) < 1e-6 or abs(mygrad1(x[-1])) < 1e-6:
            break
        x_new = x[-1] - 3*mycost1(x[-1])/mygrad1(x[-1])
        #print(x_new, cost(x[-1]), grad(x[-1]))
        x.append(x_new)
    return (x, it)
# (x, it) = GD_newton(5)
# print(x[-1], it)

#phuong phap MomenTum
def Momentum(x0, eta = 0.1, gamma = 0.9):
    v = [0]
    x = [x0]
    for it in range(100):
        g = mygrad1(x[-1])
        if abs(g) < 1e-6:
            break
        v_new = gamma*v[-1] + eta*g
        x_new = x[-1] - v_new
        v.append(v_new)
        x.append(x_new)
    return (np.asarray(x), v, it)

# (x, v, it) = Momentum(5, 0.1, 0.9)
# print(x[-1], v[-1], it)


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

# x = np.asarray(x)
# (x, v, it) = Momentum(4, 0.01, 0.09)
# (x) = GD_newton1(4)
# (x, it) = myGD21(0,0.09)
#(x1, it1) = myGD21(4, 0.09)
# print(x[-1], it)
print(x1[-1], it1)
viz_alg_1d(x, mycost1)