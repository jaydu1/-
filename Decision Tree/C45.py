#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:40:44 2017

@author: dujinhong
"""

from numpy import *
import math
import copy
import cPickle as pickle

class C45_Tree(object):
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
        
        BestFeature,FeatureValueList=self.getBestFeature(DataSet)
        BestFeatureLabels=labels[BestFeature]
        tree={BestFeatureLabels:{}}
        del(labels[BestFeature])
        
        for value in FeatureValueList:
            SubLabels=labels[:]
            SplitDataSet=self.splitDataSet(DataSet,BestFeature,value)
            SubTree=self.buildTree(SplitDataSet,SubLabels)
            tree[BestFeatureLabels][value]=SubTree
        return tree
        
    def Max(self,List):
        items=dict([(List.count(i),i) for i in List])
        return items[max(items.keys())]
 
# 选取最优特征节点
    def getBestFeature(self,DataSet):
        NumFeatures=len(DataSet[0])-1
        total=len(DataSet)
        BaseEntropy=self.computeEntropy(DataSet)
        ConditionEntropy=[]
        SplitInfo=[]
        FeatureList=[]
        
        for i in xrange(NumFeatures):
            List=[data[i] for data in DataSet]
            [Split,FeatureValueList]=self.computeSplitInfo(List)
            FeatureList.append(FeatureValueList)
            SplitInfo.append(Split)
            Gain=0.0
            for value in FeatureValueList:
                SubDataSet=self.splitDataSet(DataSet,i,value)
                appearNum=float(len(SubDataSet))
                SubEntropy=self.computeEntropy(SubDataSet)
                Gain+=(appearNum/total)*SubEntropy
            ConditionEntropy.append(Gain)
        InfoGain=BaseEntropy*ones(NumFeatures)-array(ConditionEntropy)

        InfoGainRatio=InfoGain/(SplitInfo+exp(-8))
        BestFeatureIndex=argsort(-InfoGainRatio)[0]
        return BestFeatureIndex,FeatureList[BestFeatureIndex]
    
    def computeSplitInfo(self,List):
        datalen=len(List)
        DataSet=list(set(List))
        ValueCount=[List.count(Vector) for Vector in DataSet]
        probList=[float(item)/datalen for item in ValueCount]
        iList=[item*math.log(item,2) for item in probList]
        splitInfo=-sum(iList)
        return splitInfo,DataSet
    
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