'''
XGB make predictions
generating predictions and putting into xarray takes about 5 minutes for each model
6 models takes about 30 minutes
'''
import pickle
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import sklearn
import xgboost as xgb

def load_XGB(model_path=None):
    '''
    load_XGB: loads model from path
    model_path : needs to to be .pkl file
    
    !! this could just be a general `pickle_load()` function
    '''
    # =============================
    # load the model
    # =============================
    # Load from file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


def df_to_xarray(df_in=None):
    '''
    df_to_xarray(df_in) converts dataframe to dataset
        this makes a monthly 1x1 skeleton dataframe already
        time, lat, lon need to be in the dataframe
    !! this take 4 minutes !!
    
    example
    ==========
    ds = df_to_xarray(df_in = df[['time','lat','lon','sst']])
    '''
    
    # to make date in attributes
    from datetime import date

    # Make skeleton 
    dates = pd.date_range(start=f'1982-01-01', end=f'2018-12-01',freq='MS')+ np.timedelta64(14, 'D')
    ds_skeleton = xr.Dataset({'lon':np.arange(0.5, 360, 1), 
                              'lat':np.arange(-89.5, 90, 1),
                              'time':dates})    
    # make dataframe
    skeleton = ds_skeleton.to_dataframe().reset_index()[['time','lat','lon']]

    # Merge predictions with df_all dataframe
    df_out = skeleton.merge(df_in, how = 'left', on = ['time','lat','lon'])
    
    # conver to xarray dataset
    # old way to `dimt, = ds_skeleton.time.shape` ect. to get dimensions
    # then reshape  `df_out.values.reshape(dim_lat, dim_lon, dim_time)`
    # finally create a custom dataset
    df_out.set_index(['time', 'lat','lon'], inplace=True)
    ds = df_out.to_xarray()
    #ds['sst'].attrs['units'] = 'uatm'

    return ds
    
    
def main():  
    # =============================
    # define direcotries and file names
    # =============================
    # which algorithm is this for, this is used to define 
    algorithm = 'XGB'
    
    # name of model to load
    model_dir = f'/home/gloege/projects/ldeo_hpd/models/{algorithm}/GCB_2020_plus_xco2'
    output_dir = f'/home/gloege/projects/ldeo_hpd/data/model_output/{algorithm}/GCB_2020_plus_xco2'
    
    # name of dataset to load
    data_dir = '/home/gloege/projects/ldeo_hpd/data'
    dataset = f'{data_dir}/processed/PREDICTION_mon_1x1_198201-201812.pkl'
    
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
        print(f'processing {mod} ...')
        mod_short_name = mod.split('_')[0]
        model_name = f'{algorithm}_{mod_short_name}'
        
        # ==========================================================
        # load the model
        # ==========================================================
        model_path = f'{model_dir}/{model_name}.pkl'
        model = load_XGB(model_path=f'{model_path}')

        # ==========================================================
        # Load data
        # ==========================================================
        print(f'loading data ...')
        df = pd.read_pickle(f'{dataset}')  

        # calculate model error --> uses for test/train/val sets
        #df['error'] = df['fco2'] - df[f'{mod}']

        # train / validation split
        features = ['log_chl','log_mld','sst','sss', 'xco2',
                    'log_chl_anom','sst_anom',
                    'A','B','C','T0','T1',]
        
        # --> used for test/train/val sets
        #target = ['error']

        # ==========================================================
        #  make predictions
        # for XGB this needs to be dataframe, needs to know variable names 
        # predictions take about 25 secs
        # ==========================================================
        print(f'making predictions...')
        pred = model.predict(df[features])

        # put predictions into dataframe
        df[f'error_{mod}'] = pred
        
        # add error back to model
        df[f'corrected_{mod}'] = df[f'error_{mod}'] + df[f'{mod}']

        # ==========================================================
        #  put predictions back into 
        # lat/lon space
        # this step takes about 4 minutes
        # ==========================================================
        print(f'put output into file ...')
        print(f'{output_dir}/{algorithm}_{mod}_mon_1x1_198201-201812.nc')
        ds = df_to_xarray(df[['time', 'lat', 'lon', f'error_{mod}', f'corrected_{mod}']])
        ds.to_netcdf(f'{output_dir}/{algorithm}_{mod}_mon_1x1_198201-201812.nc')
        print('Complete !')
        print('=======================================================')
        print('')
        
        
if __name__ == '__main__':
    main()