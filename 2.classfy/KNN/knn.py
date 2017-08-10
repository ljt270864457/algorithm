# coding: utf-8

from __future__ import division
import pandas as pd
from collections import Counter

class Preprossing(object):
    def __init__(self, data_path, train_percent):
        print 'data preprossing...'
        self.data_path = data_path
        self.train_percent = train_percent

    def readData(self):
        return pd.read_csv(self.data_path, header=None, sep=',')

    def cleanData(self, df):
        '''
        最后一列g:1 b:0
        所有的列的数据缩放至0-1
        '''
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 'g' else 0)
        df.iloc[:, :-1] = df.iloc[:, :-1].apply(lambda x: (x - x.min()) / (x.max() -
                      x.min()) if x.max() != x.min() else x)
        return df

    def splitData(self, df):
        train_rows_count = int(df.shape[0] * self.train_percent)
        train_x = df.iloc[:train_rows_count, :-1]
        train_y = df.iloc[:train_rows_count, -1]
        test_x = df.iloc[train_rows_count:, :-1]
        test_y = df.iloc[train_rows_count:, -1]
        return train_x, test_x, train_y, test_y


class KNN(object):
    def __init__(self, train_x, train_y, test_x, test_y, K=3):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.K = K
        self.train_rows=self.train_x.shape[0]
        self.test_rows=self.test_x.shape[0]

    def fit(self):
        '''
        使用欧氏距离进行计算,返回排序之后的距离
        '''
        print 'fiting ......'
        result_list = []       
        for i in range(self.test_rows):            
            self.train_x['distance'] = self.train_x.apply(
                lambda x: (sum((x - self.test_x.iloc[i])**2)**0.5), axis=1)
            top_K = self.train_x.sort_values(['distance']).iloc[:self.K,]
            tags = list(self.train_y[list(top_K.index)])
            result_tag = Counter(tags).most_common(1)
            result_list.append(result_tag[0][0])
        result_dataFrame = pd.DataFrame({'actual':list(self.test_y),'predict':result_list})
        return result_dataFrame

    def validate(self,df):
        new_df=df[df['actual']==df['predict']]
        correct_count = new_df.shape[0]
        accuracy = (correct_count/self.test_rows)*100
        print 'accuracy percent is {0}%'.format(round(accuracy,2))

if __name__ == '__main__':
    data_reader= Preprossing('./ionosphere.data', 0.7)
    data= data_reader.readData()
    cleanedData= data_reader.cleanData(data)
    train_x, test_x, train_y, test_y= data_reader.splitData(cleanedData)
    knn = KNN(train_x, train_y, test_x, test_y)
    df = knn.fit()
    print df
    knn.validate(df)
