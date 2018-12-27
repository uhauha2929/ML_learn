# -*- coding: utf-8 -*-
# @Time    : 2018/10/17 13:57
# @Author  : uhauha2929
import itertools
import time
from pprint import pprint

import numpy as np


class Apriori(object):

    def __init__(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_sets = []
        self.frequent_supports = {}
        self.transactions = None

    def find_frequent_itemsets(self, transactions):
        if self.frequent_sets:
            # 如果存在，直接返回，否则会重复添加
            return self.frequent_sets
        self.transactions = transactions
        unique_items = np.array(sorted(set([item for transaction in transactions for item in transaction])))
        unique_items = unique_items.reshape(-1, 1).tolist()
        frequent = self._get_frequent_itemsets(unique_items)
        self.frequent_sets.append(frequent)
        # 逐层搜索频繁项集
        while True:
            candidates = self._generate_candidates(self.frequent_sets[-1])  # 上一层的候选集
            frequent = self._get_frequent_itemsets(candidates)
            if not frequent:
                break
            self.frequent_sets.append(frequent)

        return self.frequent_sets

    def _get_frequent_itemsets(self, candidates):
        # 判断候选集是否大于最小支持度，如果是则加入频繁项集
        frequent = []
        for itemset in candidates:
            support = self._calculate_support(itemset)
            if support >= self.min_support:
                frequent.append(itemset)
                self.frequent_supports[tuple(itemset)] = support
        return frequent

    def _calculate_support(self, itemset):
        # 判断项集是否在事务中，并计算支持度
        count = 0
        for transaction in self.transactions:
            if not [False for item in itemset if item not in transaction]:
                count += 1
        support = count / len(self.transactions)
        return support

    def _generate_candidates(self, frequent_set):
        candidates = []
        l = len(frequent_set[0])
        for i in range(len(frequent_set)):
            for j in range(i + 1, len(frequent_set)):
                # 频繁项集中已经排好序，其中每一项如果前l-1个值相同，合并成后判断是否是候选项集
                if frequent_set[i][:l - 1] == frequent_set[j][:l - 1]:
                    candidate = frequent_set[i] + frequent_set[j][-1:]
                    infrequent = self._has_infrequent_itemsets(candidate)
                    if not infrequent:
                        candidates.append(candidate)

        return candidates

    def _has_infrequent_itemsets(self, candicate):
        # 如果一个集合的子集不在上一层频繁项集中，则该候选项也是不频繁的
        subsets = itertools.combinations(candicate, len(candicate) - 1)
        for subset in subsets:
            if list(subset) not in self.frequent_sets[-1]:
                return True
        return False

    def generate_rules(self, transactions):
        frequent_itemsets = self.find_frequent_itemsets(transactions)
        for itemset in frequent_itemsets[1:]:  # 忽略第一层
            for item in itemset:
                # 对每个频繁项，其子集一定在频繁项集里
                for k in range(1, len(item)):
                    for subset in itertools.combinations(item, k):
                        confidence = self.frequent_supports[tuple(item)] / self.frequent_supports[subset]
                        if confidence >= self.min_confidence:
                            print(subset, '->', set(item) - set(subset), confidence)


if __name__ == '__main__':
    transactions = [[1, 2, 3, 4], [1, 2, 4], [1, 2], [2, 3, 4], [2, 3], [3, 4], [2, 4]]
    apriori = Apriori(0.25, 0.8)
    print('Rules:')
    apriori.generate_rules(transactions)
    print('Frequent Itemsets:')
    pprint(apriori.frequent_sets)

    time_start = time.time()
    lines = []
    with open('test.txt', 'rt') as f:
        for line in f:
            lines.append(line.strip().split(','))
    apriori = Apriori(0.06, 0.75)
    apriori.generate_rules(lines)
    time_end = time.time()
    print('time cost', time_end - time_start)
