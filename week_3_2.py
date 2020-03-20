import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time

t0 = time()


df = pd.read_csv('invasion.csv', sep=',')

clf = RandomForestClassifier()


params = {'n_estimators':range(10,51,10),
          'max_depth':range(1,13,2),
          'min_samples_leaf':range(1,8),
          'min_samples_split':range(2,10,2)
          }

y_train = df['class']
X_train = df
del X_train['class']

grid = GridSearchCV(clf, params, cv=5)
grid.fit(X_train, y_train)

t1 = time()
print(t1 - t0)

best_params = grid.best_params_

def get_best_params(parameters):
    """
    accepts a dict with params
    
    Default values:
    ===============
    n_estimators: int
        10
    max_depth: int
        10
    min_sample_leaf: int
        1
    min_sample_split: int
        2
    
    Returns:
    ========
        An RandomForestClassifier object with parameters set in users dict.
    """
    estimators = parameters.get('n_estimators', 10)
    m_depth = parameters.get('max_depth', 10)
    min_samples_l = parameters.get('min_samples_leaf', 1)
    min_samples_s = parameters.get('min_samples_split', 2)
    
    clf = RandomForestClassifier(n_estimators=estimators,
                                 max_depth=m_depth,
                                 min_samples_split=min_samples_s,
                                 min_samples_leaf=min_samples_l)
    
    return(clf)
    


clf_2 = get_best_params(best_params)

X_test = pd.read_csv('operative_information.csv', sep=',')

clf_2.fit(X_train, y_train)
res = clf_2.predict(X_test)

unique, counts = np.unique(res, return_counts=True)
print(dict(zip(unique, counts)))
print(time()-t0)
