import numpy
import cvxopt
from cvxopt import matrix, solvers
from math import log, exp
# type C
# c = matrix([-6., -4., -5.])
# G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,
#                    7.,  -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.],
#                 [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,
#                    3.,  13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.],
#                 [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,
#                    9.,   6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.]])
# h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,
#                   68., -30., -19., -30.,  99.,  23., -19.,   23.,  10.] )
# dims = {'l': 2, 'q': [4, 4], 's': [3]}
# sol = solvers.conelp(c, G, h, dims)
# sol['status']
# 'optimal'
# print(sol['x'])
# print(sol['z'])

# QP method
#ex1
# P = matrix([[4.0,0.0],[0.0,0.0]])
# q = matrix([1.0,2.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 1.0], (1,2))
# b = matrix(1.0)
# solqp = solvers.qp(P,q,G,h, A, b)

# print(sol['x'])
# print(sol['primal objective'])

#ex10

# P = matrix([[4.0,-2.0,-2.],[-2.0,4.0,-    2.],[-2., -2., 4.]])
# q = matrix([0.0, 0.0, 0.])
# G = matrix([[0.0,0.0,0.],[0.0,0.0,0.0],[0.0,0.0,0.0]])
# h = matrix([0.0,0.0,0.])
# A = matrix([[1., 3.], [2., 2.], [3., 1.]])
# b = matrix([10.,14.])

# sol = solvers.qp(P,q,G,h,A,b)
# print(sol['x'])
# print(sol['primal objective'])


#Linear Method
#ex14
# c = matrix([-5.0, -3.0])
# G = matrix([[1.0, 2.0, 1.0, -1.0, 0.0], [1.0, 1.0, 4.0, 0.0, -1.0]])
# h = matrix([10.0, 16.0, 32.0, 0.0, 0.0])
# solln = solvers.lp(c, G, h)

# print(solln['x'])

#ex16

# c = matrix([1.0, -2.0, -4.0, 2.0])
# G = matrix([[1.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0,], [-2.0, 0.0, 8, 0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0]])
# h = matrix([4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 0.0])

# solln = solvers.lp(c, G, h)

# print(solln['x'])

#Geometry method
#ex
K = [1]
F = matrix([[1.],[-1.]])
g = matrix([log(1.)])
G = matrix([[0.5, 0.], [-0.33, 0.]])
h = matrix([1., 1.])

solvers.options['show_progress'] = False
solgp = solvers.gp(K, F, g)
print('Solution:')
print((solgp['x'])))
