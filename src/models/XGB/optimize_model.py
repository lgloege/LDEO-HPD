'''
Some guidelines for tuning a neural network
focus on #layers, #units, learning rate, optimizer,
===================================================================
1. choose start values
    max_depth = 5, min_child_weight = 1 
    gamma = 0 ,subsample, colsample_bytree = 0.8 
2. tune max_depth and min_child_weight
3. tune gamma
4. tune ,subsample and colsample_bytree
5. tune reg_alpha and reg_beta, if you want to

# explanation of parameters
===================
n_estimators: Number of trees in random forest
learning_rate: Makes the model more robust by shrinking the weights on each step
               Typical final values to be used: 0.01-0.2
max_depth: The maximum depth of a tree, same as GBM.
min_child_weight: Defines the minimum sum of weights of all observations required in a child.
max_leaf_node: The maximum number of terminal nodes or leaves in a tree.
gamma: A node is split only when the resulting split gives a positive reduction in the loss function. 
       Gamma specifies the minimum loss reduction required to make a split.
max_delta_step:
subsample: Denotes the fraction of observations to be randomly samples for each tree.
colsample_by_tree: Denotes the fraction of columns to be randomly samples for each tree.
colsample_by_level: Denotes the subsample ratio of columns for each split, in each level.
                    do not use if you use colsample_by_tree
lambda: L2 regularization term on weights (analogous to Ridge regression)
        This used to handle the regularization part of XGBoost. 
        not used often, but it should be explored to reduce overfitting.
alpha: L1 regularization term on weight (analogous to Lasso regression)
       Can be used in case of very high dimensionality so the algorithm runs faster 
scale_pos_weight: value > 0 should be used if high class imbalance as it helps in faster convergence.

max_features: Number of features to consider at every split ['auto', 'sqrt']
max_depth: Maximum number of levels in tree
min_samples_split: Minimum number of samples required to split a node  [2, 5, 10]
min_samples_leaf: Minimum number of samples required at each leaf node [1, 2, 4]
boostrap : Method of selecting samples for training each tree [True, False]

# References
===================
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20662
https://blog.socratesk.com/assets/pdf/Parameter_Tuning_XGBoost.pdf
https://towardsdatascience.com/from-zero-to-hero-in-xgboost-tuning-e48b59bfaf58
'''
# standard imports
import os
import scipy
import numpy as np
import pandas as pd
import xgboost as xgb

# modeling
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.model_selection import GridSearchCV

# settings 
pd.set_option("display.max_columns", 100)
def train_val_split(df=None, features=None, target=None, 
                    test_size=0.2, random_state=43):
    '''
    train validation split
    '''
    X = df.loc[:, features]
    y = df.loc[:, target]

    # Uses train_test_split build into sklearn.model_selection
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, 
                                                      shuffle=True, 
                                                      random_state=random_state)

    out_dict = {'X_train': X_train[features],
                'X_val': X_val[features],
                'y_train': y_train,
                'y_val': y_val,}

    return out_dict


def main():
    # Load data
    data_dir = '/home/gloege/projects/ldeo_hpd/data'
    df = pd.read_pickle(f'{data_dir}/processed/TRAIN_mon_1x1_198201-201812.pkl')  

    #models = [ 'cesm_spco2_1x1_A', 
    #       'recom_jra_spco2_1x1_A', 
    #       'mpi_spco2_1x1_A', 
    #       'cnrm_spco2_1x1_A',
    #       'noresm_spco2_1x1_A',
    #       'planktom_spco2_1x1_A',
    #       'princeton_spco2_1x1_A',
    #       'csiro_spco2_1x1_A',
    #       'ipsl_spco2_1x1_A']
        
    # calculate model error
    df['error'] = df['pco2'] - df['cesm_spco2_1x1_A']

    # train / validation split
    features = ['log_chl','log_mld','sst','sss',
                'log_chl_anom','sst_anom',
                'A','B','C','T0','T1',]
    target = ['error']

    # define train validaiton set
    df_dict = train_val_split(df, features=features, target=target)

    # need to set validation data
    X_val_ready = df_dict['X_val'].values
    y_val_ready = df_dict['y_val'].values

    X_train_ready = df_dict['X_train'].values
    y_train_ready = df_dict['y_train'].values
    
    # =======================================
    # define keras regressor and grid
    # =======================================
    # Use the random grid to search for best hyperparameters  
    param_best = {'objective': 'reg:squarederror',
                  'max_depth': 9, 
                  'min_child_weight': 1, 
                  'gamma': 0,
                  'subsample': 0.8,
                  'colsample_bytree': 0.9,
                  'reg_alpha': 0.09,
                  'reg_labmda': 1,
                  #'n_estimators': 1000,
                  'learning_rate': 0.05,
                 }
                  #'learning_rate': 0.05}
    
    #param_best = {'objective': 'reg:squarederror',
    #              'max_depth': 9, 
    #              'min_child_weight': 1, 
    #              'gamma': 0,
    #              'subsample': 0.85,
    #              'colsample_bytree': 0.95,
    #              'reg_alpha': 0,
    #              'reg_labmda': 1,
    #              'n_estimators': 1000,
    #              'learning_rate': 0.05}
    
    
    # fit parameters 
    # https://stackoverflow.com/questions/42993550/gridsearchcv-xgboost-early-stopping
    fit_params={"early_stopping_rounds":50, 
                "eval_metric" : "mae", 
                "eval_set" : [[X_val_ready, y_val_ready]]}
    
    model = xgb.XGBRegressor(**param_best, **fit_params)

    # parameter space to search
    param_grid={
        #'reg_alpha':[8e-2, 9e-2, 0.1, 0.11, 0.12]
        #'learning_rate':[0.25, 0.2, 0.15, 0.1, 0.05]
        # 'subsample':[i/10.0 for i in range(6,10)],
        # 'colsample_bytree':[i/10.0 for i in range(6,10)],
        # 'max_depth':[8,9,10], #range(3,10,2),
        # 'min_child_weight':[1,2,3],  #range(1,6,2),
        'n_estimators': [500, 1000, 2000, 5000]
        }

    # create grid object
    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid,
                        cv=3, 
                        scoring='neg_mean_squared_error')

    # =======================================
    # perform grid search:
    # =======================================
    # fit each parameter
    grid_result = grid.fit(X_train_ready, y_train_ready)

    # show best parameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # =======================================
    # summarize results
    # =======================================
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    
if __name__ == "__main__":
    main()
