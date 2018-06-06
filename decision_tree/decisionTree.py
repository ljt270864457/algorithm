#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午10:12
# @Author  : liujiatian
# @File    : decisionTree.py

from __future__ import division

from math import log

import pandas as pd


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    df = pd.DataFrame(data=dataSet, columns=['outlook', 'temperature', 'humidity', 'windy', 'target'])
    return df


def calcShannonEnt(dataSet):
    '''
    计算信息熵
    :return:
    '''
    entropy = 0.0
    length = len(dataSet)
    '''
    counter={'N':3,'Y':4}
    '''
    counter = dict(dataSet.iloc[:, -1].value_counts())
    for _, count in counter.iteritems():
        prob = count / length
        log_prob = log(prob, 2)
        entropy -= prob * log_prob
    return entropy


def chooseFeatureByInfoGain(dataSet):
    '''
    ID3
    根据最大信息增益选择最佳分裂维度
    :param dataSet:
    :return: 维度的序数
    '''
    if dataSet.empty:
        return

    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    featureCount = dataSet.shape[1] - 1
    dataSetRows = len(dataSet)

    # 维度的数量
    for i in range(featureCount):
        # 不同元素
        uniqueVals = set(list(dataSet.iloc[:, i]))
        splitInfo = 0.0
        for value in uniqueVals:
            subDataFrame = dataSet[dataSet.iloc[:, i] == value]
            subDataFrameRows = len(subDataFrame)
            prob = subDataFrameRows / dataSetRows
            splitInfo += prob * calcShannonEnt(subDataFrame)
        infoGain = baseEntropy - splitInfo
        print u'第%s列信息增益是：%s' % (i, infoGain)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def chooseFeatureByInfoGainRatio(dataSet):
    '''
    C4.5
    根据最大信息增益率选择最佳分裂维度
    :param dataSet:
    :return: 维度的序数
    '''
    if dataSet.empty:
        return

    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    featureCount = dataSet.shape[1] - 1
    dataSetRows = len(dataSet)

    # 维度的数量
    for i in range(featureCount):
        # 不同元素
        uniqueVals = set(list(dataSet.iloc[:, i]))
        splitInfo = 0.0
        for value in uniqueVals:
            subDataFrame = dataSet[dataSet.iloc[:, i] == value]
            subDataFrameRows = len(subDataFrame)
            prob = subDataFrameRows / dataSetRows
            splitInfo += prob * calcShannonEnt(subDataFrame)
        infoGain = baseEntropy - splitInfo
        infoGainRatio = infoGain / baseEntropy
        print u'第%s列信息增益率是：%s' % (i, infoGainRatio)
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGain
            bestFeature = i
    return bestFeature


def createTree(dataSet):
    '''
    创建决策树
    :param dataSet:
    :return: {'outlook': {0: 'N', 1: 'Y', 2: {'windy': {0: 'Y', 1: 'N'}}}}
    '''
    # 如果所有的label都一样，停止分裂
    if len(dataSet.iloc[:, -1].value_counts()) == 1:
        target = dataSet.iloc[:, -1].value_counts().index[0]
        return target
    # 如果只剩下一个维度，那么取众数
    if len(dataSet.columns) == 2:
        target = dataSet.iloc[:, -1].mode()[0]
        return target

    # ID3
    # bestFeatureIndex = chooseFeatureByInfoGain(dataSet)
    # C4.5
    bestFeatureIndex = chooseFeatureByInfoGainRatio(dataSet)
    bestFeature = dataSet.columns[bestFeatureIndex]
    myTree = {bestFeature: {}}
    for item, subDataSet in dataSet.groupby(bestFeature):
        myTree[bestFeature][item] = createTree(subDataSet)
    return myTree


if __name__ == '__main__':
    dataSet = createDataSet()
    print dataSet
    print createTree(dataSet)
