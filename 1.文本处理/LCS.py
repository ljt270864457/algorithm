# coding=utf-8

import numpy as np
import pandas as pd


class LCS(object):
    def __init__(self, strA, strB):
        self.strA = strA
        self.strB = strB
        self.lenth_A = len(strA)
        self.lenth_B = len(strB)
        self.index = map(lambda x: x, self.strA)
        self.index.insert(0, None)
        self.columns = map(lambda x: x, self.strB)
        self.columns.insert(0, None)
        self.matrix = self.__init_matrix()

    def __init_matrix(self):
        matrix = np.zeros((self.lenth_A + 1, self.lenth_B + 1), dtype=int)
        matrix = pd.DataFrame(matrix, columns=self.columns, index=self.index)
        return matrix

    def LCS(self):
        matrix = self.matrix
        for ii, char_index in enumerate(self.index):
            for jj, char_column in enumerate(self.columns):
                if None == char_column or char_index == None:
                    matrix.iloc[ii, jj] = 0
                elif char_column == char_index:
                    matrix.iloc[ii, jj] = matrix.iloc[ii - 1, jj - 1] + 1
                else:
                    matrix.iloc[ii, jj] = max(
                        matrix.iloc[ii - 1, jj], matrix.iloc[ii, jj - 1])
        return matrix, matrix.iloc[-1, -1]

if __name__ == '__main__':
    a = LCS("ABCBDAB", "BDCABA")
    lcsMatrix, maxSubCount = a.LCS()
    print '-----LCS矩阵-----'
    print lcsMatrix
    print '最大公共子序列的数量为：{0}'.format(maxSubCount)
