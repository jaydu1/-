#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:22:54 2017

@author: dujinhong
"""

from numpy import *
import math
import copy
import cPickle as pickle

class ID3_Tree(object):
    def __init__(self):
        self.tree={}
        self.DataSet=[]
        self.labels=[]
        
    def loadDataSet(self,path,labels):
        recordlist=[]
        fp=open(path,"rb")
        content=fp.read()
        fp.close()
        rowlist=content.splitlines()
        recordlist=[row.split("\t") for row in rowlist if row.strip()]
        self.DataSet=recordlist
        self.labels=labels
        
    def train(self):
        labels=copy.deepcopy(self.labels)
        self.tree=self.buildTree(self.DataSet,labels)
        
    def buildTree(self,DataSet,labels):
        List=[data[-1] for data in DataSet] # 抽取决策标签
        # 若决策标签只有一种
        if List.count(List[0])==len(List):
            return List[0]
        # 若第一个标签的数据只有一个
        if len(DataSet[0])==1:
            return self.Max(List)
        
        BestFeature=self.getBestFeature(DataSet)
        BestFeatureLabels=labels[BestFeature]
        tree={BestFeatureLabels:{}}
        del(labels[BestFeature])
        
        UniqueValue=set([data[BestFeature] for data in DataSet])
        for value in UniqueValue:
            SubLabels=labels[:]
            SplitDataSet=self.splitDataSet(DataSet,BestFeature,value)
            SubTree=self.buildTree(SplitDataSet,SubLabels)
            tree[BestFeatureLabels][value]=SubTree
        return tree
        
    def Max(self,List):
        items=dict([(List.count(i),i) for i in List])
        return items[max(items.keys())]
    
    def getBestFeature(self,DataSet):
        NumFeatures=len(DataSet[0])-1
        BaseEntropy=self.computeEntropy(DataSet)
        BestInfoGain=0.0
        BestFeature=-1;
        
        for i in xrange(NumFeatures):
            UniqueValues=set([data[i] for data in DataSet])
            NewEntropy=0.0
            for value in UniqueValues:
                SubDataSet=self.splitDataSet(DataSet,i,value)
                prob=len(SubDataSet)/float(len(DataSet))
                NewEntropy+=prob*self.computeEntropy(SubDataSet)
            InfoGain=BaseEntropy-NewEntropy
            if InfoGain>BestInfoGain:
                BestInfoGain=InfoGain
                BestFeature=i
        return BestFeature
    
    def computeEntropy(self,DataSet):
        datalen=float(len(DataSet))
        List=[data[-1] for data in DataSet]
        
        items=dict([(i,List.count(i)) for i in List])
        InfoEntropy=0.0
        for key in items:
            prob=float(items[key])/datalen
            InfoEntropy-=prob*math.log(prob,2)
        return InfoEntropy
    
    def splitDataSet(self,DataSet,axis,value):
        List=[]
        for FeatureVector in DataSet:
            if FeatureVector[axis]==value:
                RestFeatureVector=FeatureVector[:axis]
                RestFeatureVector.extend(FeatureVector[axis+1:])
                List.append(RestFeatureVector)
        return List
        
# 保存，读取决策树
    def saveTree(self,inputTree,filename):
        f=open(filename,"w")
        pickle.dump(inputTree,f)
        f.close()
        
    def getTree(self,filename):
        f=open(filename)
        return pickle.load(f)
        
# 分类器
    def predict(self,inputTree,labels,testVector):
        root=inputTree.keys()[0]
        secondDict=inputTree[root]
        Index=labels.index(root)
        key=testVector[Index]
        value=secondDict[key]
        if isinstance(value,dict):
            classLabel=self.predict(value,labels,testVector)
        else:
            classLabel=value
        return classLabel
    
    