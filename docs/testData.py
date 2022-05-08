# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:55:12 2021

@author: JennyLee
"""

from pygcn.utils import load_data

adj, features, labels, idx_train, idx_val, idx_test = load_data()

print("adj:")
print(adj)
print("\n\nfeatures:")
print(features)
print("\n\nlabels:")
print(labels)
print("\n\nidx_train:")
print(idx_train)
print("\n\nidx_val:")
print(idx_val)
print("\n\nidx_test:")
print(idx_test)
