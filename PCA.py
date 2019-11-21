#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/21 9:39 PM
# @Author  : liujiatian
# @File    : PCA.py

from __future__ import division
import numpy as np

ori_mean_matrix = np.array([[1, 1], [1, 3], [2, 3], [4, 4], [2, 4]])
sub_mean_matrix = ori_mean_matrix - np.mean(ori_mean_matrix, axis=0)
data_count = sub_mean_matrix.shape[0]

# 1. 减均值后的向量 其中X是原始数据的转置，X_T是原始数据
X = sub_mean_matrix.T
X_T = sub_mean_matrix

print X
print X_T
print data_count

# 2.获取协方差矩阵C
C = 1 / data_count * np.dot(X, X_T)
print '协方差矩阵：', C

# 3.计算协方差矩阵的特征值和特征向量

# vals:特征值 ves 特征向量
eig_vals, eig_vecs = np.linalg.eig(C)
feature_list = []
for i in range(len(eig_vals)):
    feature_list.append([eig_vals[i], eig_vecs[..., i]])

# 4.特征值排序(降序) 求topK 对应的特征向量P
K = 1
sorted_feature_list = sorted(feature_list, key=lambda x: x[0], reverse=True)
P = np.stack(map(lambda x: x[1], sorted_feature_list[:K]), axis=0)

# 5.Y就是降维后的结果
Y = np.dot(P, X)
print Y
