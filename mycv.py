import numpy as np
import pandas as pd
from scipy.special import comb
from itertools import combinations
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit
)

class CombinatorialPurgedGroupKFold():
    def __init__(self, n_splits = 6, n_test_splits = 2, purge = 1, pctEmbargo = 0.01, **kwargs):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge
        self.pctEmbargo = pctEmbargo
        
    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
            
        u, ind = np.unique(groups, return_index = True)
        unique_groups = u[np.argsort(ind)]
        n_groups = len(unique_groups)
        group_dict = {}
        for idx in range(len(X)):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
                
        n_folds = comb(self.n_splits, self.n_test_splits, exact = True)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
            
        mbrg = int(n_groups * self.pctEmbargo)
        if mbrg < 0:
            raise ValueError(
                "The number of 'embargoed' groups should not be negative")
        
        split_dict = {}
        group_test_size = n_groups // self.n_splits
        for split in range(self.n_splits):
            if split == self.n_splits - 1:
                split_dict[split] = unique_groups[int(split * group_test_size):].tolist()
            else:
                split_dict[split] = unique_groups[int(split * group_test_size):int((split + 1) * group_test_size)].tolist()
        
        for test_splits in combinations(range(self.n_splits), self.n_test_splits):
            test_groups = []
            banned_groups = []
            for split in test_splits:
                test_groups += split_dict[split]
                banned_groups += unique_groups[split_dict[split][0] - self.purge:split_dict[split][0]].tolist()
                banned_groups += unique_groups[split_dict[split][-1] + 1:split_dict[split][-1] + self.purge + mbrg + 1].tolist()
            train_groups = [i for i in unique_groups if (i not in banned_groups) and (i not in test_groups)]

            train_idx = []
            test_idx = []
            for train_group in train_groups:
                train_idx += group_dict[train_group]
            for test_group in test_groups:
                test_idx += group_dict[test_group]
            yield train_idx, test_idx

def get_CPGKfold(train, target_col, group_col, n_splits = 6, n_test_splits = 2, purge = 1, pctEmbargo = 0.01):
    fold_series = []
    kf = CombinatorialPurgedGroupKFold(n_splits=n_splits, n_test_splits=n_test_splits, purge=purge, pctEmbargo=pctEmbargo)
    kf_generator = kf.split(train, train[target_col], train[group_col])
    for fold, (idx_train, idx_valid) in enumerate(kf_generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

def get_kfold(train, n_splits, seed):
    fold_series = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf_generator = kf.split(train)
    for fold, (idx_train, idx_valid) in enumerate(kf_generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

def get_stratifiedkfold(train, target_col, n_splits, seed):
    fold_series = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf_generator = kf.split(train, train[target_col])
    for fold, (idx_train, idx_valid) in enumerate(kf_generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

def get_groupkfold(train, target_col, group_col, n_splits):
    fold_series = []
    kf = GroupKFold(n_splits=n_splits)
    kf_generator = kf.split(train, train[target_col], train[group_col])
    for fold, (idx_train, idx_valid) in enumerate(kf_generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

def get_timeseriesfold(train, n_splits):
    fold_series = []
    kf = TimeSeriesSplit(n_splits=n_splits)
    kf_generator = kf.split(train)
    for fold, (idx_train, idx_valid) in enumerate(kf_generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    fold_series = pd.concat([pd.Series(-np.ones(len(train)-len(fold_series)), dtype=np.int64), fold_series])
    return fold_series