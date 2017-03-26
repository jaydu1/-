#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:24:47 2017

@author: dujinhong
"""

# Scikit-Learn实现CART算法

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def plotfigure(X,X_test,y,yp):
    plt.figure()
    plt.scatter(X,y,c="k",label="data")
    plt.plot(X_test,yp,c="r",label="max_depth=4",linewidth=2)
    plt.xlabel("label")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
    
x=np.linspace(-5,5,200)
siny=np.sin(x)
X=np.mat(x).T
y=siny+np.random.rand(1,len(siny))*1.5
y=y.tolist()[0]

clf=DecisionTreeRegressor(max_depth=4)
clf.fit(X,y)

X_test=np.arange(-5.0,5.0,0.05)[:,np.newaxis]
yp=clf.predict(X_test)

plotfigure(X,X_test,y,yp)        