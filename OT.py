#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:30:49 2020

@author: s1998345
"""
import pylab 

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


n=10
#gauss = lambda q,a,c: a*np.random.randn(2, q) + np.transpose(np.tile(c, (q,1)))
#X = np.random.randn(2,n)*.3
#Y = np.hstack((gauss(int(m/2),.5,[0,1.6]),np.hstack((gauss(int(m/4),.3,[-1,-1]),gauss(int(m/4),.3,[1,-1])))))
a = np.ones((n, 1))
b = np.ones((n, 1))
X = np.array(np.linspace(0,1, num=n)).reshape((n, 1))
Y = np.array(np.linspace(1,2, num=n)).reshape((1, n))
print(Y)

def distmat1(x,y):
    return x**2 + y**2 - 2*x.dot(y)
C1 = distmat1(X,Y)

def distmat2(x,y):
    return abs(x - y)**0.5
C2 = distmat2(X,Y)


#set up optimization problem
P = cp.Variable((n,n))
u = np.ones((n,1))
v = np.ones((n,1))
U = [0 <= P, cp.matmul(P,u)==a, cp.matmul(P.T,v)==b]

objective = cp.Minimize( cp.sum(cp.multiply(P,C1)) )
prob = cp.Problem(objective, U)
result = prob.solve()

plt.figure(figsize = (5,5))
plt.imshow(P.value)
plt.savefig('/Optimal transport.png')


