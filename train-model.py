import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
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
y1 = np.ones(x1.shape[0])

x = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1), axis=0)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 14000)
# m = 1000

# x_train, x_test = x[:m], x[m:]
# y_train, y_test = y[:m], y[m:]
model = LogisticRegression(C=1e5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(100*acc)

joblib.dump(model, "train_number.py", compress=3)
#print((100*accuracy_score(y_test, y_pred.tolist()))

# plt.imshow(x_all.T[:,40064].reshape(28,28))
# plt.axis('off')
# plt.show()
