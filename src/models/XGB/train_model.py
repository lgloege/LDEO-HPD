'''
Train XGB models

- takes about 3 minutes to train
- 6 models takes about 20 minutes to train

THIS RUN WAS CHANGED TO INCLUDE XCO2 asa predictor

'''
# standard imports
import pickle
import os
import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split


def train_val_split(df=None, features=None, target=None, test_size=0.2, random_state=43):
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
    algorithm = 'XGB'
    model_dir = f'/home/gloege/projects/ldeo_hpd/models/{algorithm}/GCB_2020_plus_xco2'
    
    #models = list(df.keys()[df.keys().str.contains('spco2_')])
    #models = ['spco2_CCSM', 'spco2_MPI', 'spco2_NEMO', 'spco2_NORESM', 'spco2_RECOM']    
    models = [ 'cesm_spco2_1x1_A', 
               'csiro_spco2_1x1_A',
               'fesom_spco2_1x1_A',
               'mpi_spco2_1x1_A', 
               'cnrm_spco2_1x1_A',
               'ipsl_spco2_1x1_A',
               'planktom_spco2_1x1_A',
               'noresm_spco2_1x1_A',
               'princeton_spco2_1x1_A',]
    
    for mod in models:
        print(mod)
        mod_short_name = mod.split('_')[0]
        model_name = f'{algorithm}_{mod_short_name}'
        
        # =======================================
        # Load data
        # =======================================
        data_dir = '/home/gloege/projects/ldeo_hpd/data'
        df = pd.read_pickle(f'{data_dir}/processed/TRAIN_mon_1x1_198201-201812.pkl')  

        # calculate model error
        df['error'] = df['pco2'] - df[f'{mod}']

        # train / validation split
        features = ['log_chl','log_mld','sst','sss','xco2',
                    'log_chl_anom','sst_anom',
                    'A','B','C','T0','T1',]
        target = ['error']

        df_dict = train_val_split(df, features=features, target=target)

        # =======================================
        # define model
        # =======================================
        #param_best = {'random_state':42,
        #              'objective': 'reg:squarederror',  
        #              'max_depth': 9, 
        #              'min_child_weight': 1, 
        #              'gamma': 0,
        #              'subsample': 0.85,
        #              'colsample_bytree': 0.95,
        #              'reg_alpha': 0,
        #              'reg_labmda': 1,
        #              'n_estimators': 1000,
        #              'learning_rate': 0.05}

        param_best = {'objective': 'reg:squarederror',
                      'random_state':42,
                      'max_depth': 9, 
                      'min_child_weight': 1, 
                      'gamma': 0,
                      'subsample': 0.85,
                      'colsample_bytree': 0.95,
                      'reg_alpha': 0.09,
                      'reg_labmda': 1,
                      'n_estimators': 1500,
                      'learning_rate': 0.05,
                     }


        # =========================================
        # 7. Model training
        # =========================================
        print('model training')
        model = xgb.XGBRegressor(**param_best)

        # Train the model
        model.fit(df_dict['X_train'], np.ravel(df_dict['y_train']))

        # =======================================
        # save model architecture and weights
        # =======================================
        # Save to file in the current working directory
        pkl_filename = f"{model_dir}/{model_name}.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

    print('Complete !')
    
if __name__ == '__main__':
    main()