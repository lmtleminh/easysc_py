# -*- coding: utf-8 -*-
"""
This is an easy binning solution for credit scorecard build.

Author : Tri Le <lmtleminh@gmail.com>

Categorical binning for Credit Scoring

This is an easy categorical binning solution for credit scorecard build. It is designed to
group the optimal categories by utilizing the 
which is only applied on factor variables.

"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib import parallel_backend
import sys
from matplotlib import pyplot as plt
import optuna

class ScBase:
    def __init__(self, n_iter = 10, n_jobs = None, p = .03, min_rate = .05, 
                 threshold = .0, best = True, outlier = True, 
                 missing_values = None, random_state = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.p = p
        self.min_rate = min_rate
        self.threshold = threshold
        self.best = best   
        self.outlier = outlier
        self.missing_values = missing_values
        self.random_state = random_state

class ScBinning:
    """
    
    """
    def __init__(self, n_iter = 10, n_jobs = None, p = .03, min_rate = .05, 
                 threshold = .0, best = True, plot = False, outlier = True, 
                 missing_values = None, random_state = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.p = p
        self.min_rate = min_rate
        self.threshold = threshold
        self.best = best
        self.plot = plot
        self.outlier = outlier
        self.missing_values = missing_values
        self.random_state = random_state
     
    def _woe(self, X, y):
        b = pd.Series(y.reshape(1, -1)[0], name = 'BAD')
        g = pd.Series((1-y).reshape(1, -1)[0], name = 'GOOD')
        X_x = pd.concat((X, b, g), axis = 1).groupby(X.name).agg('sum')
        pct_b = (X_x / b.sum()).values[:,0]
        pct_g = (X_x / g.sum()).values[:,1]
        woe = np.log(pct_g / pct_b)
        iv = (pct_g - pct_b) * woe
        return (woe, iv, np.array(X_x.index))
    
    def _superbin(self, X, y):
        def _optimal_bin(n_cl, p, min_rate):
            k_md = KMeans(n_clusters = n_cl, random_state = self.random_state)
            k_md.fit(thres.reshape(-1, 1))
            thres_g = k_md.predict(thres.reshape(-1, 1))
            
            new_thres = np.zeros(np.unique(thres_g).max() + 1, dtype = 'float64')
            for i in np.unique(thres_g):
                new_thres[i] = (thres[thres_g == i] * cnt_thres[thres_g == i]).sum() / cnt_thres[thres_g == i].sum()
            new_thres.sort()
            new_thres = np.append(new_thres, np.inf)
            
            i = 0
            n = train_X.shape[0]
            pre_b_r = .0
            while i < len(new_thres):
                if len(new_thres) <= 2:
                    break  
                lower_b = (train_X > new_thres[i-1]) if i > 0 else (np.ones(n) == 1).reshape(-1, 1)
                upper_b = (train_X <= new_thres[i])
                if (lower_b & upper_b).sum()/n < p:
                    new_thres = np.delete(new_thres, i if i < (len(new_thres) - 1) else i - 1)
                    i = 0
                elif np.abs(train_y[(lower_b & upper_b)].mean() - pre_b_r) < min_rate:
                    new_thres = np.delete(new_thres, i if i < (len(new_thres) - 1) else i - 1)
                    i = 0
                else:
                    pre_b_r = train_y[(lower_b & upper_b)].mean()
                    i += 1
            
            new_thres = np.append(new_thres, -np.inf)
            new_thres.sort()
            
            X_x = pd.cut(pd.Series(val_X.reshape(1, -1)[0], name = 'X'), new_thres, labels = new_thres[1:]).astype('float64')
            return self._woe(X_x, val_y)
        
        train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = .8,
                                                          random_state = self.random_state,
                                                          stratify = y)
               
        #set seed
        np.random.seed(self.random_state)
        #bootstrap sampling
        idx = np.arange(0, train_X.shape[0])
        
        idx_bt = np.concatenate((np.random.choice(idx, (len(idx), self.n_iter), 
                                                  replace = True), idx.reshape(-1,1)), axis = 1)
        
        #run tree in parallel
        with parallel_backend('threading', n_jobs = self.n_jobs):
            thresholds = Parallel()(delayed(self._treeBin)(train_X[ix], train_y[ix]) for ix in idx_bt.T)
        
        #calculate bad rate
        thresholds = np.hstack(thresholds)
        thresholds.sort()
        thres, cnt_thres = np.unique(thresholds, return_counts = True)
        
        #kmeans
        def objective(trial):
            n_cl = trial.suggest_int('n_clusters', 2,  min(10, thres.reshape(-1, 1).shape[0]))
            if isinstance(self.p, list):
                p = trial.suggest_uniform('p', min(self.p), max(self.p))
            elif isinstance(self.p, float):
                p = self.p
            if isinstance(self.min_rate, list):
                min_rate = trial.suggest_uniform('min_rate', min(self.min_rate), max(self.min_rate))
            elif isinstance(self.min_rate, float):
                min_rate = self.min_rate
            iv = _optimal_bin(n_cl, p, min_rate)[1].sum()
            return iv
        
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective, n_trials = 20)

        woe, iv, thres_1 = _optimal_bin(study.best_params['n_clusters'],
                                        study.best_params['p'] if isinstance(self.p, list) else self.p,
                                        study.best_params['min_rate'] if isinstance(self.min_rate, list) else self.min_rate)
        
        #isotonic
        # #trend
        # ins = np.polyfit(thres, bad_rate[:-1], 1)[0]
       
        # #prepare iso table
        # iso_t = np.hstack((np.append(thres, thres.max() + 1).reshape(-1,1), 
        #                    bad_rate.reshape(-1,1)))
        # iso_t = np.repeat(iso_t, np.append(cnt_thres, 1), axis = 0)
        
        # #iso regression
        # ir = IsotonicRegression(increasing = (ins >= 0))
        # bad_rate_fit = ir.fit_transform(iso_t[:,0], iso_t[:,1])
        # thresholds_t = np.hstack((iso_t[:,0].reshape(-1,1), bad_rate_fit.reshape(-1,1)))
        
        # if self.plot:
        #     fig = plt.figure()
        #     plt.plot(iso_t[:,0], iso_t[:,1], 'r.')
        #     plt.plot(iso_t[:,0], bad_rate_fit, 'b-')
        #     plt.show()
        
        # thres_1 = np.array([thresholds_t[thresholds_t[:,1] == x, 0].mean() for x in np.unique(thresholds_t[:,1])])
        # thres_1.sort()
        
        # j = 0
        # while j <= len(thres_1):
        #     if len(thres_1) <= 1:
        #         print('Minimum number of bins encountered!\n')
        #         break
        #     if j == 0:
        #         bad_rate_1 = np.zeros(len(thres_1) + 1, dtype = 'float64')
        #         cnt_1 = np.zeros(len(thres_1) + 1, dtype = 'int64')
        #         bad_rate_1[j] = y[X <= thres_1[j]].mean()
        #         cnt_1[j] = len(y[X <= thres_1[j]])
        #     elif j < len(thres_1):
        #         bad_rate_1[j] = y[(X <= thres_1[j]) & (X > thres_1[j-1])].mean()
        #         cnt_1[j] = len(y[(X <= thres_1[j]) & (X > thres_1[j-1])])
        #     else:
        #         bad_rate_1[j] = y[X > thres_1[j-1]].mean()
        #         cnt_1[j] = len(y[X > thres_1[j-1]])
            
        #     if j >= 0:
        #         if (abs(bad_rate_1[j] - bad_rate_1[j-1]) < self.min_rate/100) | (cnt_1[j] / len(X) < self.p / 100): #self
        #             thres_1 = np.delete(thres_1, j-1 if j == len(thres_1) else j)
        #             j = 0
        #         else:
        #             j += 1
        #     else:
        #         j += 1     
        
        # k = 0
        # X_tt = np.zeros(len(X), dtype = 'int').reshape(-1,1)
        # while k <= len(thres_1):
        #     if k == 0:
        #         X_tt = X_tt + np.where(X <= thres_1[k], k, 0)
        #     elif k < len(thres_1):
        #         X_tt = X_tt + np.where((X <= thres_1[k]) & (X > thres_1[k-1]), k, 0)
        #     else:
        #         X_tt = X_tt + np.where(X > thres_1[k-1], k, 0)
        #     k += 1
            
        # thres_1 = np.append(thres_1, np.inf)
        # thres_1.sort()
    
        # woe, iv, col_ = self._woe(X_tt, y)
        
        col_ = np.arange(len(thres_1))
        return np.vstack((col_, thres_1, woe)), iv.sum()
                       
    def _treeBin(self, X, y):  
        clf = DecisionTreeClassifier(max_depth = 2, random_state = self.random_state)
        clf.fit(X, y)
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        is_leaves = np.zeros(shape=clf.tree_.node_count, dtype=bool)
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()
            if (children_left[node_id] != children_right[node_id]):
                stack.append(children_left[node_id])
                stack.append(children_right[node_id])
            else:
                is_leaves[node_id] = True
        return threshold[~is_leaves]
    
    def _prefit(self, X, y, i):
        X_t = X.copy()
        y_t = y
        if self.missing_values is not None:
            if isinstance(self.missing_values, list):
                m_v = self.missing_values
            else:
                m_v = [self.missing_values]
            m_out = ~X_t.isin(m_v).values
            X_t = X_t[m_out]
            y_t = y_t[m_out]
        if self.outlier:
            Q1 = X_t.quantile(.25)
            Q3 = X_t.quantile(.75)
            IQR = Q3 - Q1
            q_out = ~((X_t < (Q1 - 1.5 * IQR)) | (X_t > (Q3 + 1.5 * IQR)))
            X_t = X_t[q_out]
            y_t = y_t[q_out]
        try:
            X_t = VarianceThreshold(self.threshold).fit_transform(X_t.values.reshape(-1,1))
        except ValueError:
            sys.stdout.write(' No feature in X meets the variance threshold {}\n'.format(self.threshold))
        else:
            try:
                self.thres_[i], self.iv_[i] = self._superbin(X_t, y_t)
                self.columns_.append(i)
            except (IndexError, ValueError):
                sys.stdout.write(' No optimal splits\n')
            else:
                if self.missing_values is not None:
                    l = 0
                    while l < len(m_v):
                        if m_v[l] not in X.values:
                            m_v = np.delete(m_v, l)
                        else:
                            l += 1  
                    X_t = np.zeros(len(X), dtype = float)
                    X_tt = X.values
                    for j in range(self.thres_[i].shape[1]):
                        if j == 0:
                            X_t = X_t + np.where(X_tt <= self.thres_[i][1][j], self.thres_[i][0][j], 0)
                        else:
                            X_t = X_t + np.where((X_tt <= self.thres_[i][1][j]) & (X_tt > self.thres_[i][1][j-1]), 
                                    self.thres_[i][0][j], 0)
                    
                    thres_1 = self.thres_[i][1]
                    for l in range(len(m_v)):
                        X_t[X_tt == m_v[l]] = -(l + 1) #m_v[l]
                        thres_1 = np.append(m_v[l], thres_1)
                    
                    woe, iv, col_ = self._woe(pd.Series(X_t.reshape(1, -1)[0], name = 'X'), y)
                
                    thres_1 = np.vstack((col_, thres_1, woe))
             
                    self.thres_[i], self.iv_[i] = thres_1, iv.sum()
        return self
    
    
    def fit(self, X, y):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
        
        y = check_array(y)
        self.thres_ = {}
        self.n_features_ = 1
        self.columns_ = []
        self.iv_ = {}
        if isinstance(X, pd.core.frame.DataFrame):
            n = 0
            self.n_features_ = X.shape[1]
            for i in X:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, n+1, X.shape[1]))
                sys.stdout.flush()
                n += 1
                if X[i].dtype == 'object':
                    sys.stdout.write('Categorical values, try ScCatBinning instead!\n')
                    continue
                else:
                    self._prefit(X[i], y, i)
                
        if isinstance(X, pd.core.series.Series):
            sys.stdout.write('Processing : %s, %s out of %s.\n' % (X.name, 1, 1))
            sys.stdout.flush()
            if X.dtype != 'object':
                self._prefit(X, y, X.name)
                
                                
        if isinstance(X, np.ndarray):
            try:
                X.shape[1] > 1
            except IndexError:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                self._prefit(pd.Series(X), y, 0)
                
            else:
                self.n_features_ = len(X.T)
                for i in range(len(X.T)):
                    sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, i+1, len(X.T)))
                    sys.stdout.flush()
                    self._prefit(pd.Series(X[:,i]), y, i)
                    
        sys.stdout.write('Done! \n')
        return self
    
    def _prepredict(self, X, i, k):            
        if self.missing_values is not None:
            if isinstance(self.missing_values, list):
                m_v = self.missing_values
            else:
                m_v = [self.missing_values]
            l = 0
            while l < len(m_v):
                if m_v[l] not in X.values:
                    m_v = np.delete(m_v, l)
                else:
                    l += 1
        else:
            m_v = []
        X_t = np.zeros(len(X), dtype = float).reshape(-1, 1)
        X_tt = X.values.reshape(-1,1)
        for j in range(self.thres_[i].shape[1]):
            if (j == 0) & (self.thres_[i][1][j] not in m_v):
                X_t = X_t + np.where(X_tt <= self.thres_[i][1][j], self.thres_[i][k][j], 0)
            elif self.thres_[i][1][j] not in m_v:
                X_t = X_t + np.where((X_tt <= self.thres_[i][1][j]) & \
                                     (X_tt > (self.thres_[i][1][j-1] if self.thres_[i][1][j-1] not in m_v else -np.inf)), 
                                     self.thres_[i][k][j], 0)
        for j in range(self.thres_[i].shape[1]):
            if self.thres_[i][1][j] in m_v:
                X_t[X_tt == self.thres_[i][1][j]] = self.thres_[i][k][j]
        return X_t 
        
    
    def predict(self, X, types = 'woe', return_all_col = False):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                          pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
    
        X_s = X.copy()
        if isinstance(X, pd.core.frame.DataFrame):
            if self.n_features_ != X.shape[1]:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is %s and "
                                 "input n_features is %s "
                                 % (self.n_features_, X.shape[1]))
            for i in X:
                if i in self.columns_:
                    X_s[i] = self._prepredict(X[i], i, 0 if types == 'category' else 2)
            if ~return_all_col:
                X_s = X_s[self.columns_]
                    
    
        if isinstance(X, pd.core.series.Series):
        
            if X.name in self.thres_.keys():
                X_s = self._prepredict(X, X.name, 0 if types == 'category' else 2)
                
        if isinstance(X, np.ndarray):
            try:
                X.shape[1] > 1
            except:
                X_s = self._prepredict(pd.Series(X), 0, 0 if types == 'category' else 2).reshape(1, -1)
            else:
                if self.n_features_ != len(X.T):
                    raise ValueError("Number of features of the model must "
                                     "match the input. Model n_features is %s and "
                                     "input n_features is %s "
                                     % (self.n_features_, len(X.T)))
                for i in range(len(X.T)):
                    if i in self.columns_:
                        X_s[:,i] = self._prepredict(pd.Series(X[:,i]), i, 0 if types == 'category' else 2).reshape(1, -1)
                if ~return_all_col:
                    X_s = X_s[:,self.columns_]
        return X_s
        
class ScCatBinning:
    """
    
    """
    def __init__(self, n_iter = 10, n_jobs = None, p = .03, min_rate = .05, 
                 threshold = 0, best = True, random_state = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.p = p
        self.min_rate = min_rate
        self.threshold = threshold
        self.best = best   
        self.random_state = random_state

    def _woe(self, X, y):
        pct_b = [y[X == j].sum() for j in np.unique(X)] / y.sum()
        pct_g = [(1-y[X == j]).sum() for j in np.unique(X)] / (1-y).sum()
        woe = np.log(pct_g / pct_b)
        iv = (pct_g - pct_b) * woe
        return (woe, iv, np.unique(X)) 

    def _impactCode(self, X, y):
        
        u_levels = np.append(np.unique(X), '__NULL__')
        
        def _catwisemean(X, y):
            mean = y.mean()
            smooth = 300
            m_ = np.array([(y[X == i].sum() + smooth * mean)/(len(y[X == i]) + smooth)
                      for i in u_levels])
            return m_
        
        #set seed
        np.random.seed(self.random_state)
        #bootstrap sampling
        idx = np.arange(0, X.shape[0])
        
        idx_bt = np.concatenate((np.random.choice(idx, 
                                               (len(idx), self.n_iter
                                                ), replace = True), idx.reshape(-1,1)), axis = 1)
        
        #run tree in parallel
        with parallel_backend('threading', n_jobs = self.n_jobs):
            m_totals = Parallel()(delayed(_catwisemean)(X[ix], y[ix]) for ix in idx_bt.T)
        
        #element wise mean
        m = np.mean(m_totals, axis = 0)
        
        #set number of optimal group
        n = 8 if len(u_levels) >= 9 else len(u_levels)-1
        b = True
        #Kmeans
        while b:
          km_m = KMeans(n_clusters = n, random_state = self.random_state)
          km_m.fit(m.reshape(-1,1))

          thres = np.vstack((km_m.labels_, u_levels))
          X_t = np.zeros(len(X), dtype = int).reshape(-1, 1)
          for i in range(thres.shape[1]):
            X_t = X_t + (np.where(X == thres[1,i], thres[0][i], 0))
        
          if np.any(np.unique(X_t, return_counts= True)[1] / len(X_t) < self.p): 
            n -= 1
          else:
            b = False
        
        woe, iv, labels = self._woe(X_t, y)
        woe_label = np.vstack((woe, labels))
        woe_transform = np.zeros(thres.shape[1], dtype = float)
        for j in range(woe_label.shape[1]):
            woe_transform = woe_transform + (np.where(thres[0] == woe_label[1, j], 
                                                      woe_label[0, j], 0))
        thres = np.vstack((thres, woe_transform))
        return thres, iv.sum()

    def fit(self, X, y):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
      
        y = check_array(y)
        self.thres_ = {}
        self.n_features_ = 1
        self.columns_ = []
        self.iv_ = {}
        if isinstance(X, pd.core.frame.DataFrame):
            n = 0
            self.n_features_ = X.shape[1]
            for i in X:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, n+1, X.shape[1]))
                sys.stdout.flush()
                n += 1
                if X[i].dtype != 'object':
                    continue
                else:
                    try:
                        X_t = X[i].values.reshape(-1,1)#VarianceThreshold(self.threshold).fit_transform(X[i].values.reshape(-1,1))
                        self.thres_[i], self.iv_[i] = self._impactCode(X_t, y)
                        self.columns_.append(i)
                    except:
                        sys.stdout.write(' No feature in X meets the variance threshold 0.00000\n')
                        continue
                
        if isinstance(X, pd.core.series.Series):
            sys.stdout.write('Processing : %s, %s out of %s.\n' % (X.name, 1, 1))
            sys.stdout.flush()
            if X.dtype == 'object':
                X_t = X.values.reshape(-1,1)#VarianceThreshold(self.threshold).fit_transform(X.values.reshape(-1,1))
                self.thres_[X.name], self.iv_[X.name] = self._impactCode(X_t, y)
        if isinstance(X, np.ndarray):
            try:
                X.shape[1] == 1
            except:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                raise ValueError('Expected 2D array, got 1D array instead:\
                                 Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.')
            if X.shape[1] == 1:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                try:
                    X = check_array(X)
                    sys.stdout.write('Numeric values, try ScBinning instead!\n')
                except:
                    X_t = X #VarianceThreshold(self.threshold).fit_transform(X)
                    self.thres_[0], self.iv_[0] = self._impactCode(X_t, y)
            else:
                self.n_features_ = len(X.T)
                for i in range(len(X.T)):
                    sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, i+1, len(X.T)))
                    sys.stdout.flush()
                    try:
                        X_t = check_array(X[:,i].reshape(-1,1))
                        sys.stdout.write('Numeric values, try ScBinning instead!\n')
                    except:
                        self.thres_[i], self.iv_[i] = self._impactCode(X[:,i].reshape(-1,1), y)
        sys.stdout.write('Done! \n')
        return self             
    
    def predict_woe(self, X, impute_missing = True):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
        
        X_s = X.copy()
        if isinstance(X, pd.core.frame.DataFrame):
            if self.n_features_ != X.shape[1]:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is %s and "
                                 "input n_features is %s "
                                 % (self.n_features_, X.shape[1]))
            for i in X:
                if i in self.columns_:
                    X_t = np.zeros(len(X[i]), dtype = float).reshape(-1, 1)
                    X_tt = X[i].values.reshape(-1,1)
                    for j in range(self.thres_[i].shape[1]):
                        X_t = X_t + (np.where(X_tt == self.thres_[i][1,j], 
                                          self.thres_[i][2][j], 0))
                    if impute_missing:
                        X_t[X_t == 0] = self.thres_[i][2][j]
                        
                    X_s[i] = X_t
        
        if isinstance(X, pd.core.series.Series):
            
            if X.name in self.thres_.keys():
                X_t = np.zeros(len(X), dtype = float)
                X_tt = X.values
                for j in range(self.thres_[X.name].shape[1]):
                    X_t = X_t + (np.where(X_tt == self.thres_[X.name][1,j],
                                          self.thres_[X.name][2][j], 0))
                if impute_missing:
                    X_t[X_t == 0] = self.thres_[X.name][2][j]
                
                X_s = pd.Series(X_t)
        if isinstance(X, np.ndarray):
            try:
                X.shape[1] == 1
            except:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                raise ValueError('Expected 2D array, got 1D array instead:\
                                 Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.')
            if X.shape[1] == 1:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                try:
                    X = check_array(X)
                    sys.stdout.write('Numeric values, try ScBinning instead!\n')
                except:
                    X_t = np.zeros(len(X), dtype = float).reshape(-1,1)
                    for j in range(self.thres_[0].shape[1]):
                        X_t = X_t + (np.where(X == self.thres_[0][1,j],
                                              self.thres_[0][2][j], 0))
                    if impute_missing:
                        X_t[X_t == 0] = self.thres_[0][2][j]
                    X_s = X_t
            else:
                if self.n_features_ != len(X.T):
                    raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is %s and "
                                 "input n_features is %s "
                                 % (self.n_features_, len(X.T)))
                for i in range(len(X.T)):
                    try:
                        X_t = check_array(X[:,i].reshape(-1,1))
                        sys.stdout.write('Numeric values, try ScBinning instead!\n')
                    except:
                        X_t = np.zeros(len(X[:,i]), dtype = float)
                        for j in range(self.thres_[i].shape[1]):
                            X_t = X_t + (np.where(X[:,i] == self.thres_[i][1,j],
                                              self.thres_[i][2][j], 0))
                        if impute_missing:
                            X_t[X_t == 0] = self.thres_[i][2][j]
                        X_s[:,i] = X_t.flatten()
        return X_s
    
    def update(self, X, y, var_name, group):
        self.thres_[var_name][0] = group
        X_t0 = np.zeros(len(X[var_name]), dtype = int).reshape(-1, 1)
        for i in range(self.thres_[var_name].shape[1]):
            X_t0 = X_t0 + (np.where(X[var_name] == self.thres_[var_name][1,i], 
                                    self.thres_[var_name][0][i], 0)).reshape(-1,1)
        woe_, iv_, group_ =  self._woe(X_t0, y)
        woe_t = np.zeros(self.thres_[var_name].shape[1])
        for i in range(len(group_)):
            woe_t = woe_t + np.where(self.thres_[var_name][0] == group_[i], woe_[i], 0)
        self.thres_[var_name][2] = woe_t
        self.iv_[var_name] = iv_.sum()
        return self
