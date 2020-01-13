import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
#Load data set
mnist = fetch_mldata('mnist-original',data_home = './')
N,d = mnist.data.shape
print(N)
print(d)
x_all = mnist.data
y_all = mnist.target
print(y_all)
#print(x_all)    
#SHow a number in dataset

x0 = x_all[np.where(y_all == 0)[0]]
x1 = x_all[np.where(y_all == 1)[0]]

y0 = np.zeros(x0.shape[0])
y1 = np.zeros(x1.shape[1])

x = np.concatenate((x0, x1), axis=0)
y = np.concatenate(y0, y1)


plt.imshow(x_all.T[:,12664].reshape(28,28))
plt.axis('off')
plt.show()