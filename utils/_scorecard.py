# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:05:39 2019

@author: trilm3
"""
import pandas as pd
import numpy as np
from ..binning import ScBinning
from ..binning import ScCatBinning

class ScScore:
    """
    
    """
    def __init__(self, model, pdo = 50, score = 500, odd = 2, *args):
        self.model = model
        self.args = args
        #self.ScB = bin_num
        #self.ScBcat = bin_cat
        self.pdo = pdo
        self.score = score
        self.odd = odd
        self.point_ = {}
    
    def _point(self, X, m, factor, offset):
        if len(self.args) > 0:
            k = 0
            for j in self.args:
                k += (isinstance(j, ScCatBinning) | isinstance(j, ScBinning))
            if k > 0:
                i = 0
                for j in X:
                    #print(j)
                    thres = None
                    for l in self.args:
                        try:
                            thres = l.thres_[j]
                        except:
                            continue
                    if not thres is None:
                        self.point_[j] = np.vstack((thres[1], 
                                   np.round((self.model.coef_[0,i] * thres[2].astype('float') + 
                                             (self.model.intercept_/m)) * factor + (offset/m), 0)))
                    else:
                        break
                    i += 1
        
    def transform(self, X):
        X_t = pd.DataFrame()
        factor = self.pdo/np.log(2)
        offset = self.score - (factor * np.log(self.odd))
        m = self.model.coef_.shape[1]
        i = 0
        for j in X:
            X_t[j] = np.round((self.model.coef_[0,i] * X[j] +  
           (self.model.intercept_/m)) * factor + (offset/m), 0)
            i += 1
        X_t.set_index(X.index, inplace = True)
        if len(self.point_) == 0:
            self._point(X, m, factor, offset)
        return X_t

