# -*- coding: utf-8 -*-
"""
@author: huma1003
"""
import numpy as np

class Base:
    def __init__(self, base):
        self.base = []      
        dim = len(base)
        col = len(base[0])
        for i in range(dim):
            for j in range(1, col):
                self.base.append(base[i, j])
        self.base = np.array(self.base).T
