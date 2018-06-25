#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 下午8:57
# @Author  : liujiatian
# @File    : gd.py

from __future__ import division
import os
import random
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIC_DIR = os.path.join(CURRENT_DIR, 'pics')


def genRandomPoint():
    '''
    模拟y=5*x+3
    :return:
    '''
    x_list = []
    y_list = []
    for i in range(50):
        x = random.random() * 5
        y = 5 * x + 3 + random.uniform(-0.5, 0.5)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def gradientDecent(x_list, y_list):
    '''
    梯度下降
    y=ax+b
    cost_function = (1/2*N)*sum(h(i)-y(i))**2
    :return: a,b
    '''
    a = 1.0
    b = 1.0
    # 迭代次数
    iterCount = 1000
    # 学习率
    alpha = 0.001
    pointCount = len(x_list)
    costDict = {}

    assert len(x_list) == len(y_list)

    for i in range(iterCount):
        delta_a = 0.0
        delta_b = 0.0
        costFunctionValue = 0.0
        for j in range(pointCount):
            delta_a += a * x_list[j] + b - y_list[j]
            delta_b += a * x_list[j] + b - y_list[j]
            costFunctionValue += (x_list[j] * a + b - y_list[j]) ** 2
        costFunctionValue = costFunctionValue / 2 / pointCount
        costDict[i] = costFunctionValue
        delta_a = delta_a / pointCount * x_list[j]
        delta_b = delta_b / pointCount
        a -= delta_a * alpha
        b -= delta_b * alpha

    return a, b, costDict


def draw(x_list, y_list, a, b):
    plt.scatter(x_list, y_list)
    x_min = min(x_list)
    x_max = max(x_list)
    new_x_list = np.linspace(x_min, x_max, 1000)
    new_y_list = [a * x + b for x in new_x_list]
    plt.plot(new_x_list, new_y_list, 'r')
    plt.title('y={}x+{}'.format(round(a, 4), round(b, 4)))
    plt.savefig(os.path.join(PIC_DIR, 'points.png'))
    plt.close()


def drawCostFunc(costDict):
    iterCountList = costDict.keys()
    costValue = costDict.values()
    plt.plot(iterCountList, costValue)
    plt.xlabel('iterCount')
    plt.ylabel('costValue')
    plt.savefig(os.path.join(PIC_DIR, 'costFunc.png'))
    plt.close()


if __name__ == '__main__':
    x_list, y_list = genRandomPoint()
    a, b, costDict = gradientDecent(x_list, y_list)
    draw(x_list, y_list, a, b)
    drawCostFunc(costDict)
