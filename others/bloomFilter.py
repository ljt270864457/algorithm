# coding=utf-8

from __future__ import division
import math
from sympy import *
import mmh3
from bitarray import bitarray


class CalcBloomParams(object):
    '''[计算布隆过滤器参数]

    [description]

    Variables:
        c {[type]} -- [description]
        bitSize {[type]} -- [位数组的大小]
        hashCount {[type]} -- [哈希函数数量]
    '''

    def __init__(self, count, error_rate):
        self.count = count
        self.error_rate = error_rate
        self.bitArraySize = self.getBitArraySize()

    def getHashCount(self):
        x = symbols('x')
        rate = round(self.count / self.bitArraySize, 2)
        result = solve(math.e**(-x * rate) - 0.5, x)
        return int(round(result[0], 0))

    def getBitArraySize(self):
        return int(round(self.count * math.log(math.e, 2) * math.log(1 / self.error_rate, 2), 0))


class BloomFilter(object):
    def __init__(self, bitSize, hashCount):
        self.bitSize = bitSize
        self.hashCount = hashCount
        bit_array = bitarray(self.bitSize)
        bit_array.setall(0)
        self.bit_array = bit_array

    def add(self, string):
        position_list = self.get_positions(string)
        for position in position_list:
            self.bit_array[position] = 1

    def judge(self, string):
        position_list = self.get_positions(string)
        result = True
        for position in position_list:
            if self.bit_array[position] == 0:
                result = False
                break
        return result

    def get_positions(self, string):
        position_list = map(lambda x: mmh3.hash(
            string, 40 + x) % self.bitSize, range(self.hashCount))
        return position_list

    def bit2mb(self):
        return round(self.bitSize / (2**23), 2)


if __name__ == '__main__':
    dataCount = 5 * 10 ** 8
    error_rate = 0.0001
    c = CalcBloomParams(dataCount, error_rate)
    bitSize = c.getBitArraySize()
    hashCount = c.getHashCount()
    bloomFilter = BloomFilter(bitSize, hashCount)
    bloomFilter.add('http://www.baidu.com')
    print u'{0}条数据，{1}个哈希函数，误差率:{2}%,布隆过滤器的位数组大小为{3}MB'.format(dataCount, hashCount, error_rate * 100, bloomFilter.bit2mb())
    print bloomFilter.judge('http://www.baidu.com')
