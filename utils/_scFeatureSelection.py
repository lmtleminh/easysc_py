"""
This is an easy binning solution for credit scorecard build.

Author : Tri Le <lmtleminh@gmail.com>

Feature Selection for Credit Scoring

This is an easy feature selection procedure in which many methods are run in order to 
determine which subset of variables are best predicting the outcome.

"""
import time
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, roc_curve
import optuna
from joblib import Parallel, delayed
from joblib import parallel_backend
#from sklearn.utils import check_array
import numpy as np
import pandas as pd

class scFeatureSelection:
    """
    """
    def __init__(self, methods = ['median', 'chi2', 'mutual_info', 'p_cor', 's_cor', 
                                  'logreg', 'l1logreg', 'svm', 'gini_rf'], 
                 n_jobs = None, threshold_cor = 0,
                 random_state = None):
        self.methods = methods
        self.n_jobs = n_jobs
        self.threshold_cor = threshold_cor
        self.random_state = random_state

    def median(self, X, y):
        classes = np.unique(y)
        p_value = np.ones(X.shape[1])
        for i in range(X.shape[1]):
            try:
                p_val = mannwhitneyu(X[(y == classes[0]).reshape(1, -1)[0], i],
                                     X[(y == classes[1]).reshape(1, -1)[0], i],
                                     alternative = 'two-sided')[1]
            except ValueError:
                p_val = 1.0
            p_value[i] = p_val
        p_value[p_value > 5e-2] = 1.0
        p_value = np.abs(-np.log10(p_value))
        return p_value/p_value.max()
    
    def mutl_info(self, X, y):
        dp_coef = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            coef = mutual_info_classif(X[:, i].reshape(-1,1), y,
                                       random_state = self.random_state)[0]
            dp_coef[i] = coef
        return dp_coef/dp_coef.max()
    
    def chi2(self, X, y):
        p_value = np.ones(X.shape[1])
        for i in range(X.shape[1]):
            p_val = chi2(X[:, i].reshape(-1,1), y)[1][0]
            p_value[i] = p_val
        p_value[p_value > 5e-2] = 1.0
        p_value[p_value == .0] = 1.0
        p_value = np.abs(-np.log10(p_value))
        return p_value/p_value.max()
    
    def cor_func(self, X, y, type = 'p_cor', threshold = 0):
        if type == 'p_cor':
            rho = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
        elif type == 's_cor':
            rho = np.array([spearmanr(X[:, i], y)[0] for i in range(X.shape[1])])
    
        rho = np.abs(rho)
        pos = np.arange(X.shape[1])

        rho_f = rho.copy()
        pos_f = pos.copy()

        pos = pos[rho > threshold]
        rho = rho[rho > threshold]

        pos = pos[rho.argsort()]
        rho = rho[rho.argsort()]

        del_list = np.array([])
        for i in np.arange(len(pos)-1):
            for j in np.arange(i + 1, len(pos)):
                if type == 'p_cor':
                    rho_i_j = pearsonr(X[:, pos[-(i + 1)]], X[:, pos[-(j + 1)]])[0]
                elif type == 's_cor':
                    rho_i_j = spearmanr(X[:, pos[-(i + 1)]], X[:, pos[-(j + 1)]])[0]
                if rho_i_j >= rho[-(i + 1)]:
                    del_list = np.append(del_list, pos[-(j + 1)])

        pos_r = pos[np.isin(pos, del_list, invert = True)]
        rho_f[np.isin(pos_f, pos_r, invert = True)] = .0
        return rho_f/rho_f.max()
    
    def logreg_svm(self, X, y, l1 = False, svm = False):
        X = RobustScaler().fit_transform(X)
        def objective(trial):
            c = trial.suggest_uniform('c', .0, 1.0)
            if svm:
                #https://link.springer.com/content/pdf/10.1023/A:1012487302797.pdf
                clf = LinearSVC(C = c, penalty = 'l1', dual = False, 
                                class_weight = 'balanced', 
                                max_iter = 10000, random_state = self.random_state)
            else:
                clf = LogisticRegression(C = c, penalty = 'l1' if l1 else 'l2',
                                         solver = 'saga' if l1 else 'lbfgs',
                                         max_iter = 10000, random_state = self.random_state)
            clf.fit(X, y)
            if svm:
                score = clf.score(X, y)
            else:
                fpr, tpr, thres = roc_curve(y, clf.predict_proba(X)[:, 0],
                                            pos_label = 0)
            return -score if svm else -(2 * auc(fpr, tpr) - 1)

        study = optuna.create_study()
        study.optimize(objective, n_trials = 50)
        if svm:
            clf = LinearSVC(C = study.best_params['c'], penalty = 'l1',
                            dual = False, class_weight = 'balanced',
                            max_iter = 10000, random_state = self.random_state)
        else:
            clf = LogisticRegression(C = study.best_params['c'],
                                     penalty = 'l1' if l1 else 'l2',
                                     solver = 'saga' if l1 else 'lbfgs',
                                     max_iter = 10000, random_state = self.random_state)
        clf.fit(X, y)
        coef = np.square(clf.coef_[0])
        return coef/coef.max()
    
    def rf(self, X, y):
        forest = ExtraTreesClassifier(n_estimators = max(100, 2 * X.shape[1]),
                                     class_weight = 'balanced',
                                     n_jobs = self.n_jobs, random_state = self.random_state)    
        forest.fit(X, y)
        imp = forest.feature_importances_
        return imp/imp.max()

    def fit(self, X, y):
        start_time = time.perf_counter()
        optuna.logging.disable_default_handler()
        self.result = {}

        if not isinstance(X, (pd.core.frame.DataFrame,
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
      
        if isinstance(y, (pd.core.frame.DataFrame, pd.core.series.Series)):
            y = y.values
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns_ = X.columns.values
            X = X.values
        elif isinstance(X, np.ndarray):
            self.columns_ = np.arange(X.shape[0])
        if 'median' in self.methods:
            m = self.median(X, y)
            self.result['median'] = {}
            for i, na in enumerate(self.columns_):
                self.result['median'][na] = m[i]
        if 'chi2' in self.methods:
            if (X < 0).sum().sum() == 0:
                f = self.chi2(X, y)
                self.result['chi2'] = {}
                for i, na in enumerate(self.columns_):
                    self.result['chi2'][na] = f[i]
            else:
                print('Input X must be non-negative. "chi2" wont be run.\n')
        if 'mutual_info' in self.methods:
            if (X < 0).sum().sum() == 0:
                f = self.mutl_info(X, y)
                self.result['mutual_info'] = {}
                for i, na in enumerate(self.columns_):
                    self.result['mutual_info'][na] = f[i]
            else:
                print('Input X must be non-negative. "mutual_info" wont be run.\n')          
        if 'p_cor' in self.methods:
            p = self.cor_func(X, y, type = 'p_cor', threshold = self.threshold_cor)
            self.result['p_cor'] = {}
            for i, na in enumerate(self.columns_):
                self.result['p_cor'][na] = p[i]
        if 's_cor' in self.methods:
            s = self.cor_func(X, y, type = 's_cor', threshold = self.threshold_cor)
            self.result['s_cor'] = {}
            for i, na in enumerate(self.columns_):
                self.result['s_cor'][na] = s[i]
        if 'gini_rf' in self.methods:
            rf = self.rf(X, y)
            self.result['gini_rf'] = {}
            for i, na in enumerate(self.columns_):
                self.result['gini_rf'][na] = rf[i]
                
        def jobs(method):
            if method == 'logreg':
                l = self.logreg_svm(X, y)
                self.result['logreg'] = {}
                for i, na in enumerate(self.columns_):
                    self.result['logreg'][na] = l[i]
            if method == 'l1logreg':
                l_1 = self.logreg_svm(X, y, l1 = True)
                self.result['l1logreg'] = {}
                for i, na in enumerate(self.columns_):
                    self.result['l1logreg'][na] = l_1[i]
            if method == 'svm':
                sv = self.logreg_svm(X, y, svm = True)
                self.result['svm'] = {}
                for i, na in enumerate(self.columns_):
                    self.result['svm'][na] = sv[i]
            return None
                
        #run jobs in parallel
        with parallel_backend('threading', n_jobs = self.n_jobs):
            Parallel()(delayed(jobs)(method) for method in [i for i in self.methods 
                                                            if i not in ['median', 'chi2', 'mutual_info', 
                                                                         'p_cor', 's_cor', 'gini_rf']])
        
        print('Elapsed time: {:.1f} seconds'.format(time.perf_counter() - start_time))
        return self