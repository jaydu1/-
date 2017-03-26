#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:56:40 2017

@author: dujinhong
"""

#from ID3 import *
#
#labels=["age","reveue","student","credit"]
#dtree=ID3_Tree()
#dtree.loadDataSet("dataset.dat",labels)
#dtree.train()
#print dtree.tree
#
#vector=['0','1','0','0']
#print dtree.predict(dtree.tree,labels,vector)
#
#import treePlotter as tp
#tp.createPlot(dtree.tree)

from C45 import *

labels=["age","reveue","student","credit"]
dtree=C45_Tree()
dtree.loadDataSet("dataset.dat",labels)
dtree.train()
print dtree.tree

vector=['0','1','0','0']
print dtree.predict(dtree.tree,labels,vector)

import treePlotter as tp
tp.createPlot(dtree.tree)