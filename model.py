from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBRegressor as xgbr

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

#We now define the dictionary as depicted below...
model_params = {
    'LinearRegression':{
        'model': LinearRegression(),
        'params':{
            'fit_intercept': [False, True],
            'normalize': [False, True],
            'copy_X': [False, True]
        }
    },
    'Lasso':{
        'model': Lasso(),
        'params':{
            'alpha': [1,2,3,4,5,10],
            'tol': [1e-2,1e-4,1e-6,1e-8,1e-9,1e-10],
            'max_iter': [100,500,1000],
            'selection':['cyclic', 'random']
        }
    },
    'Ridge':{
        'model': Ridge(),
        'params':{
            'alpha': [1,2,3,4,5,10],
            'tol': [1e-2,1e-4,1e-6,1e-8,1e-9,1e-10],
            'max_iter': [100,500,1000]
        }
    },
    'XGBoostRegressor':{
        'model': xgbr(),
        'params':{
             "learning_rate"    : [0.05, 0.15, 0.20, 0.30, 0.4],
             "max_depth"        : [ 3, 5, 8, 12, 15],
             "min_child_weight" : [ 3, 5, 7 ],
             "gamma"            : [ 0.0, 0.1, 0.2 ],
             "colsample_bytree" : [ 0.3, 0.4, 0.7 ]
        }
    },
    'DecisionTreeClassifier':{
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['auto', 'sqrt', 'log2']
         }
     },
    'GausssianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-09, 1e-11, 1e-12]
        }
    },
    'MultinomialNB': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [1,2,3,4,5,10],
            'fit_prior': ['false', 'true']
        }
    }
}
# scores_rscv =[]
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def predictor(inputs, target):
    result_rscv = []
    scores_rscv = []  
    for model_name, mp in model_params.items():
        rscv_clf = RandomizedSearchCV(mp['model'], mp['params'], 
                                cv = 3,n_iter = 5,n_jobs = -1,
                                verbose = 3, return_train_score = False)

        start_time = timer(None)
        rscv_clf.fit(inputs,target.values.ravel())
        timer(start_time)

        scores_rscv.append({
            'Model Name': model_name,
            'Best Score': rscv_clf.best_score_,
            'Best Parameter': rscv_clf.best_params_,
            'Best Estimator': rscv_clf.best_estimator_,
            'Best Re-fit time':  rscv_clf.refit_time_
        })
        print(f'Length----------{len(scores_rscv)}.')
    result_rscv = pd.DataFrame(scores_rscv, columns = ['Model Name', 'Best Score', 'Best Parameter', 'Best Estimator', 'Best Re-fit time'])
    return result_rscv

def init(inputsFile, targetFile):
    inputs = pd.read_csv(inputsFile)
    target = pd.read_csv(targetFile)
    result  = predictor(inputs, target)
    return result.to_html()