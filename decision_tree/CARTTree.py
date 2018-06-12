#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 下午7:48
# @Author  : liujiatian
# @File    : CARTTree.py

from __future__ import division

import os
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, 'data')


def createDataSet():
    irisData = load_iris()
    data = irisData.data
    dataSet = pd.DataFrame(data, columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])
    dataSet['label'] = irisData.target
    return dataSet


def eda(dataSet):
    dataSet.iloc[:, :-1].hist()
    plt.savefig(os.path.join(DATA_PATH, 'hist.png'))
    dataSet.iloc[:, :-1].plot(kind='kde')
    plt.savefig(os.path.join(DATA_PATH, 'kde.png'))
    dataSet.iloc[:, :-1].plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.savefig(os.path.join(DATA_PATH, 'box.png'))


def calculateGini(dataSet):
    '''
    计算数据集基尼系数
    1-(p1)**2-(p2)**2
    :param dataSet:
    :return:
    '''
    rowsCount = len(dataSet)
    counter = dict(dataSet.iloc[:, -1].value_counts())
    prob = 0.0
    for label, count in counter.iteritems():
        prob += (count / rowsCount) ** 2
    gini = 1 - prob
    return gini


def chooseBestFeatureAndSplit(dataSet):
    '''
    1.遍历数据集的每个维度，并对每个维度的值进行去重排序
    2.分别切割每个维度和每个值，求出最小的giniSplit
    3.确定需要选择的维度及切割值
    :param dataSet:
    :return:
    '''
    bestFeatureIndex = -1
    bestSplitValue = 0
    bestGini = 1.0
    columns = len(dataSet.columns) - 1
    rowsCount = len(dataSet)
    for featureIndex in range(columns):
        # 对数据做去重及排序
        SortedData = list(dataSet.iloc[:, featureIndex].drop_duplicates().sort_values())

        for splitTime in range(len(SortedData) - 1):
            # 取平均值做为分割点
            splitValue = (SortedData[splitTime] + SortedData[splitTime + 1]) / 2
            df_small = dataSet[dataSet.iloc[:, featureIndex] <= splitValue]
            df_large = dataSet[dataSet.iloc[:, featureIndex] > splitValue]
            giniSplit = len(df_small) / rowsCount * calculateGini(df_small) + len(df_large) / rowsCount * calculateGini(
                df_large)
            if giniSplit <= bestGini:
                bestFeatureIndex = featureIndex
                bestSplitValue = splitValue
                bestGini = giniSplit
    return bestFeatureIndex, bestSplitValue, bestGini


def createTree(dataSet):
    bestFeatureIndex, bestSplitValue, bestGini = chooseBestFeatureAndSplit(dataSet)
    bestFeature = dataSet.columns[bestFeatureIndex]
    cartTree = {bestFeature: {}}
    df_gini = calculateGini(dataSet)
    if df_gini == 0 or len(dataSet) < 10:
        result = {
            'data': list(dataSet.iloc[:, -1].values),
            'gini': df_gini
        }
        return result
    else:
        df_left = dataSet[dataSet.iloc[:, bestFeatureIndex] <= bestSplitValue]
        df_right = dataSet[dataSet.iloc[:, bestFeatureIndex] > bestSplitValue]
        cartTree[bestFeature]['<={}'.format(bestSplitValue)] = createTree(df_left)
        cartTree[bestFeature]['>{}'.format(bestSplitValue)] = createTree(df_right)
    return cartTree


if __name__ == '__main__':
    dataSet = createDataSet()
    print createTree(dataSet)
