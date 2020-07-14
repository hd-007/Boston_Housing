# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:35:49 2020

@author: UKL
"""
import warnings
warnings.filterwarnings("ignore", module = "sklearn")

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit


# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
import matplotlib.pyplot as plt

# Load the Boston housing dataset
data = pd.read_csv('data.csv', sep='\t')

price = data['MEDV']
features = data.drop('MEDV', axis = 1)

# Success
print("Boston housing dataset has {0} data points with variables {1} each.".format(*data.shape))

#stats of price
min_price= min(price)
max_price= max(price)
mean_price=price.mean()
median_price=price.median()
std_price=price.std()

#performance_metrics
from sklearn.metrics import r2_score


def metrics_function(y_true,y_predict):
    r2score=r2_score(y_true,y_predict)
    return r2score

#Splitting_Data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(features,price)

#Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


#Random_crossValidation
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

train_sizes = np.rint(np.linspace(1,features.shape[0]*0.8-1,num=9)).astype(int)

fig = plt.figure(figsize=(20,10))

#trees_of_different_depths
for K,depth in enumerate ([1,2,3,4]):
    
    """
    
    max_depth2 = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores2, test_scores2 = validation_curve(DecisionTreeRegressor(), X_train, y_train, \
        param_name = "max_depth", param_range = max_depth2, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean2 = np.mean(train_scores2, axis=1)
    train_std2 = np.std(train_scores2, axis=1)
    test_mean2 = np.mean(test_scores2, axis=1)
    test_std2 = np.std(test_scores2, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth2, train_mean2, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth2, test_mean2, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth2, train_mean2 - train_std2, \
        train_mean2 + train_std2, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth2, test_mean2 - test_std2, \
        test_mean2 + test_std2, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()
    """
    regressor=DecisionTreeRegressor(max_depth=depth) 
    sizes,train_score,test_score=learning_curve(regressor,features,price,train_sizes=train_sizes,cv=cv)
    
    train_std = np.std(train_score, axis = 1)
    train_mean = np.mean(train_score, axis = 1)
    test_std = np.std(test_score, axis = 1)
    test_mean = np.mean(test_score, axis = 1)
    
    ax = fig.add_subplot(2, 2, K+1)
    ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
    ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
    ax.fill_between(sizes, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    ax.fill_between(sizes, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
                                                                                
    # Labels
    ax.set_title('max_depth = %s'%(depth))
    ax.set_xlabel('Number of Training Points')
    ax.set_ylabel('Score')
    ax.set_xlim([0, features.shape[0]*0.8])
    ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


#For Confirmation of results(i.e. max_depth=4) from (Validation/Complexity)/(Learning) Curves we'll use Grid search Method 

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

dt_range = range(1, 11)
params = dict(max_depth=dt_range)

# TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
# We initially created performance_metric using R2_score
scoring_fnc = make_scorer(metrics_function)

# TODO: Create the grid search object
# You would realize we manually created each, including scoring_func using R^2
grid = GridSearchCV(regressor, params, cv=cv, scoring=scoring_fnc)

# Fit the grid search object to the data to compute the optimal model
grid = grid.fit(X_train, y_train)

# Return the optimal model after fitting the data
print(grid.best_estimator_)







    