import scipy
import numpy as np
import pandas as pd
import xarray as xr
from skimage.filters import sobel
import sklearn
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#===============================================
# repeaters
#================================================

def repeat_lat_and_lon(ds=None):
    lon = np.arange(0.5,360,1)
    lat = np.arange(-89.5,90,1)
    ds_bc = xr.DataArray(np.zeros([len(lon),len(lat)]), coords=[('lon', lon),('lat', lat)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_lon(ds=None):
    lon = np.arange(0.5,360,1)
    ds_bc = xr.DataArray(np.zeros([len(lon)]), coords=[('lon', lon)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_lat(ds=None):
    lat = np.arange(-89.5,90,1)
    ds_bc = xr.DataArray(np.zeros([len(lat)]), coords=[('lat', lat)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_time(ds=None, dates=None):
    ''' dates needs to be a pandas date_range
    Example
    dates = pd.date_range(start='1982-01-01T00:00:00.000000000', 
                      end='2017-12-01T00:00:00.000000000',freq='MS')+ np.timedelta64(14, 'D')
    '''
    ds_bc = xr.DataArray(np.zeros([len(dates)]), coords=[('time', dates)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_time_and_lon(ds=None, dates=None):
    ''' dates needs to be a pandas date_range
    Example
    dates = pd.date_range(start='1998-01-01T00:00:00.000000000', 
                      end='2017-12-01T00:00:00.000000000',freq='MS')+ np.timedelta64(14, 'D')
    '''
    lon = np.arange(0.5,360,1)
    ds_bc = xr.DataArray(np.zeros([len(dates), len(lon), ]), coords=[('time', dates),('lon', lon)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data


#===============================================
# transformations
#================================================

# Define functions
def transform_doy(obj, dim='time'):
    '''
    transform_doy(ds, dim='time')
    transform DOY into repeating cycles
    
    reference
    ==========
    Gregor et al. 2019 
    '''
    obj['T0'] = np.cos(obj[f'{dim}.dayofyear'] * 2 * np.pi / 365)
    obj['T1'] = np.sin(obj[f'{dim}.dayofyear'] * 2 * np.pi / 365)
    return obj[['T0','T1']]

def compute_n_vector(obj, dim_lon='lon', dim_lat='lat'):
    '''
    compute_n_vector(ds,dim_lon='lon', dim_lat='lat')
    calculate n-vector from lat/lon
    
    reference
    ==========
    Gregor et al. 2019 
    '''
    ### convert lat/lon to radians
    obj['lat_rad'], obj['lon_rad'] = np.radians(obj[dim_lat]), np.radians(obj[dim_lon])

    ### Calculate n-vector
    obj['A'], obj['B'], obj['C'] = np.sin(obj['lat_rad']), \
                            np.sin(obj['lon_rad'])*np.cos(obj['lat_rad']), \
                            -np.cos(obj['lon_rad'])*np.cos(obj['lat_rad'])
    return obj[['A','B','C']]


#===============================================
# detrend function
#================================================


def detrend_ufunc(y):
    """
    detrend_ufunc : detrend along the first axis
    
    This takes numpy array and detrends along the first axis
    This is applied to xarray datasets using .apply_ufunc()
    
    Not as efficient as it could be. You should only detrend
    at points that are not all nan think about masking 
    mask = ~xr.ufuncs.isnan(ds['SST'])
    """

    ### Get dimensions
    ndim0 = np.shape(y)[0]
    ndim1 = np.shape(y)[1]

    ### Allocate space to store data
    y_dt = np.ones((ndim0, ndim1))*np.NaN
    #slope = np.ones((ndim1))*np.NaN
    #intercept = np.ones((ndim1))*np.NaN

    ### x vector
    x = np.arange(ndim0)

    ### Remove linear trend
    for dim1 in range(ndim1):
        ### only proceed if no NaNs
        if(np.sum(np.isnan(y[:, dim1]))==0):
            ### fit linear regression
            reg = scipy.stats.linregress(x, y[:, dim1])
            #y_dt[:,dim1] = scipy.signal.detrend(X[mask], axis=0, type='linear')
            ### make a linear fit
            yfit = reg.intercept + reg.slope * x

            ### Save regression coefficients
            #slope[dim1] = reg.slope
            #intercept[dim1] = reg.intercept

            ### subtract linear trend
            y_dt[:, dim1] = y[:, dim1] - yfit

    return y_dt

def detrend(data, dim=None):
    """
    detrend : detrends along dimension
    
    Inputs:
    ==============
    data : dataarray
    dim  : dimension to detrend over
    
    Returns
    ==============
    returns a detrended datadarray 
    
    !!!! YOU MAY NEED TO TRANSPOSE THE DIMENSIONS !!!!!
    
    Example
    ==============
    out = detrend(ds['pCO2'], dim='time')
    
    """
    ### get coordinate names 
    ### List of coords with the one you want popped out 
    coords = list(dict(data.coords).keys())
    ### Save a copy of the original coordnates for transposing purposes 
    coords_copy = coords.copy()
    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the dimensions you are not
    ### Detrending over
    ### This is so you only have one loop in detrend_ufunc
    ### We unstack there at the end
    data = data.stack({'z':coords})
    
    ### move that dimension to the beginning
    #data = data.transpose(dim, coords[0], coords[1])
    ### apply detrending function, this assumes 
    out = xr.apply_ufunc(detrend_ufunc, data)
                          #input_core_dims=[[dim]], # if you have two inputs [[dim], [dim]]
                         # vectorize=True, # !Important!
                         # output_dtypes=[float])

    ### Could not find an efficient way to transpose coords
    ### Without assuming the number the coordinates....
    ### Really slow
    #return tmp1.stack({'z':coords_copy}).unstack('z')
    ### assumes you only have 3 coordsinates
    #return out.unstack('z').transpose(coords_copy[0], coords_copy[1], coords_copy[2])
    return out.unstack('z')


#===============================================
# Masks
#================================================

def network_mask():
    '''network_mask
    This masks out regions in the 
    NCEP land-sea mask (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
    to define the open ocean. Regions removed include:
    - Coast : defined by sobel filter
    - Batymetry less than 100m
    - Arctic ocean : defined as North of 79N
    - Hudson Bay
    - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
    '''
    ### Load obs directory
    dir_obs = '/local/data/artemis/observations'
    
    ### topography
    ds_topo = xr.open_dataset(f'{dir_obs}/GEBCO_2014/processed/GEBCO_2014_1x1_global.nc')
    ds_topo = ds_topo.roll(lon=180, roll_coords='lon')
    ds_topo['lon'] = np.arange(0.5, 360, 1)

    ### Loads grids
    # land-sea mask
    # land=0, sea=1
    ds_lsmask = xr.open_dataset(f'{dir_obs}/masks/originals/lsmask.nc').sortby('lat').squeeze().drop('time')
    data = ds_lsmask['mask'].where(ds_lsmask['mask']==1)
    ### Define Latitude and Longitude
    lon = ds_lsmask['lon']
    lat = ds_lsmask['lat']
    
    ### Remove coastal points, defined by sobel edge detection
    coast = (sobel(ds_lsmask['mask'])>0)
    data = data.where(coast==0)
    
    ### Remove shallow sea, less than 100m
    ### This picks out the Solomon islands and Somoa
    data = data.where(ds_topo['Height']<-100)
    
    ### remove arctic
    data = data.where(~((lat>79)))
    data = data.where(~((lat>67) & (lat<80) & (lon>20) & (lon<180)))
    data = data.where(~((lat>67) & (lat<80) & (lon>-180+360) & (lon<-100+360)))

    ### remove caspian sea, black sea, mediterranean sea, and baltic sea
    data = data.where(~((lat>24) & (lat<70) & (lon>14) & (lon<70)))
    
    ### remove hudson bay
    data = data.where(~((lat>50) & (lat<70) & (lon>-100+360) & (lon<-70+360)))
    data = data.where(~((lat>70) & (lat<80) & (lon>-130+360) & (lon<-80+360)))
    
    ### Remove Red sea
    data = data.where(~((lat>10) & (lat<25) & (lon>10) & (lon<45)))
    data = data.where(~((lat>20) & (lat<50) & (lon>0) & (lon<20)))
    
    return data


def load_socat_mask():
    '''
    load a mask of SOCAT data product
    '''
    # location of masked data
    dir_mask = '/local/data/artemis/observations/masks/processed'
    
    # load data with xarray
    ds_mask = xr.open_dataset(f'{dir_mask}/SOCATv2019_mask_1x1_198201-201512.nc')
    
    return ds_mask


#===============================================
# Data loaders
#================================================


def load_cesm_hindcast(dir_data = '/local/data/artemis/observations/CESM_hindcast/processed',
                       date_start=None,
                       date_end=None):
    '''
    load_gcb_model(dir_gcb = '/local/data/artemis/observations/GCB_hindcast_models/processed',
                   date_start=None,
                   date_end=None)
                   
    inputs
    ============
    dir_data : directory where data files are stored
    date_start : start date for the files
    date_end : ending date for the files
    
    output
    =============
    ds : dataset with CESM hindcast trimmed to date_start:date_end
    
    notes
    ============
    could be improved. !!need to use clim. MLD and should use chl like obs. cloud cover 
    '''
    # dictionary of hindcast output
    dict_cesm = {'chl'   : f'{dir_data}/CESM-hindcast_chl_1x1_198201-201512.nc',
                 'spco2' : f'{dir_data}/CESM-hindcast_spco2_1x1_198201-201512.nc',
                 'sss'   : f'{dir_data}/CESM-hindcast_sss_1x1_198201-201512.nc',
                 'sst'   : f'{dir_data}/CESM-hindcast_sst_1x1_198201-201512.nc',
                 'xco2'  : f'{dir_data}/CESM-hindcast_xco2_1x1_198201-201512.nc',
                 'mld'   : f'{dir_data}/CESM-hindcast_mld_1x1_198201-201512.nc'}

    # load model
    ds_merge = xr.merge([
            xr.open_dataset(dict_cesm['spco2']),
            xr.open_dataset(dict_cesm['xco2']),
            xr.open_dataset(dict_cesm['chl']),
            xr.open_dataset(dict_cesm['mld']),
            xr.open_dataset(dict_cesm['sst']),
            xr.open_dataset(dict_cesm['sss'])])
    
    # Trim years
    ds_mod = ds_merge.sel(time=slice(date_start, date_end))
    
    # masking the data
    #ds_mod_train = ds_mod.where((ds_socatmask['mask']==True) & (ds_sommask['mask']==True)) 
    #ds_mod_oppo = ds_mod.where((ds_socatmask['mask']!=True) | (ds_sommask['mask']!=True)) ## rest of the data
    #ds_out = ds * ds_mask['mask']
    
    return ds_mod


def load_satellite_date(dir_data='/local/data/artemis/observations/neural_net_data'):
    '''
    load_satellite_date(dir_data='/local/data/artemis/observations/neural_net_data')
        loads a dataset with satellite observations
    '''
    # Observations dictionary
    dict_data = {'spco2': f'{dir_data}/spco2_1x1_mon_SOCATv2019_199801-201712.nc', 
                 'sst': f'{dir_data}/sst_1x1_mon_NOAAOIv2_199801-201712.nc',
                 'sss': f'{dir_data}/sss_1x1_mon_EN421_199801-201712.nc',
                 'chl': f'{dir_data}/chl_1x1_mon_globColour_199801-201712.nc',
                 'mld': f'{dir_data}/mld_1x1_clim_deBoyer_199801-201712.nc',
                 'xco2': f'{dir_data}/xco2_1x1_mon_globalview_199801-201712.nc'}

    ###  Load all variables into common dataset
    ds_obs = xr.merge([xr.open_dataset(dict_data['sst']),
                       xr.open_dataset(dict_data['sss']),
                       xr.open_dataset(dict_data['chl']),
                       xr.open_dataset(dict_data['mld']),
                       xr.open_dataset(dict_data['xco2']),
                       xr.open_dataset(dict_data['spco2'])])
    
    return ds_obs




def load_inputs(dir_data='/local/data/artemis/observations/neural_net_data/inputs_1982_2017'):
    '''
    load_satellite_date(dir_data='/local/data/artemis/observations/neural_net_data')
        loads a dataset with satellite observations
    '''
    # Observations dictionary
    dict_data = {'spco2': f'{dir_data}/spco2_1x1_mon_SOCATv2019_198201-201712.nc', 
                 'sst': f'{dir_data}/sst_1x1_mon_NOAAOIv2_198201-201712.nc',
                 'sss': f'{dir_data}/sss_1x1_mon_EN421_198201-201712.nc',
                 'chl': f'{dir_data}/chl_1x1_mon_globColour_198201-201712.nc',
                 'mld': f'{dir_data}/mld_1x1_clim_deBoyer_198201-201712.nc',
                 'xco2': f'{dir_data}/xco2_1x1_mon_globalview_198201-201712.nc'}

    ###  Load all variables into common dataset
    ds_obs = xr.merge([xr.open_dataset(dict_data['sst']),
                       xr.open_dataset(dict_data['sss']),
                       xr.open_dataset(dict_data['chl']),
                       xr.open_dataset(dict_data['mld']),
                       xr.open_dataset(dict_data['xco2']),
                       xr.open_dataset(dict_data['spco2'])])
    
    return ds_obs



def load_gcb_model(model=None,
                   dir_gcb = '/home/gloege/projects/ldeo_hpd/data/simulations',
                   date_start=None,
                   date_end=None):
    '''
    load_gcb_model(mofdel=None, 
                   dir_gcb = '/local/data/artemis/observations/GCB_hindcast_models/processed')
        loads the spcifiied GCP model:
        - CCSM
        - CNRM
        - MPI
        - NEMO
        - NORESM
        - RECOM
    '''
    # GCB dictionary
    dict_gcb = {'CCSM': f'{dir_gcb}/CCSM-BEC_spco2_1x1_195802-201801.nc', 
                'CNRM': f'{dir_gcb}/CNRM-ESM2_spco2_1x1_184801-201712.nc',
                'MPI': f'{dir_gcb}/MPI_spco2_1x1_195901-201712.nc',
                'NEMO': f'{dir_gcb}/NEMO-planktom_spco2_1x1_195901-201712.nc',
                'NORESM': f'{dir_gcb}/NorESM_spco2_1x1_194801-201712.nc',
                'RECOM': f'{dir_gcb}/REcoM_jra_spco2_1x1_195802-201801.nc'}

    # load model
    ds_mod_full = xr.open_dataset(dict_gcb[model])

    # Trim years
    ds_mod = ds_mod_full.sel(time=slice(date_start, date_end))
    #ds_mod_train = ds_mod.where((ds_socatmask['mask']==True) & (ds_sommask['mask']==True)) 
    #ds_mod_oppo = ds_mod.where((ds_socatmask['mask']!=True) | (ds_sommask['mask']!=True)) ## rest of the data
    
    return ds_mod




#===============================================
# Create features
#================================================


def create_features(ds_mod=None, ds_obs=None, dates=None):
    '''
    create_features(ds_mod=None, ds_obs=None)
        this creates input feature and outputs as a dateset
    '''
    
    # Creates a model error feature if 
    if ds_mod is not None:
        # Calculate model error
        #ds_error = (ds_mod['spco2'] - ds_hind_soc['pCO2']).rename('model_error')
        ds_error = (ds_mod['spco2'] - ds_obs['spco2']).rename('model_error')

        # Add error to dataset
        ds_obs = xr.merge([ds_obs, ds_error])

    # make sure variables are in [time, lat, lon ] order
    ds_obs = ds_obs.transpose('time', 'lat', 'lon')

    # trimmed output
    ds_trim = xr.merge([ds_obs, ## Standard variables
                        detrend(ds_obs['sst'], dim='time').rename('sst_detrend'),
                        repeat_lat_and_lon( transform_doy(ds_obs) ),           
                        repeat_time_and_lon( compute_n_vector(ds_obs)['A'] , dates=dates),
                        repeat_time( compute_n_vector(ds_obs)[['B','C']], dates=dates)])  

    # remove unnecessary variable names
    ds_trim  = ds_trim.transpose('time','lat','lon').drop(['lat_rad','lon_rad'])

    return ds_trim



#===============================================
# Train test split
#================================================


def train_val_test_split(df=None, features=None, target=None):
    ### Features (These will be features to the input)
    X = df.loc[:, features]

    ### Target (This is what the network is trying to learn)
    y = df.loc[:, target]

    ### Uses train_test_split build into sklearn.model_selection
    ### By default this method shuffles the data (30% = testing 70%=training/validation)
    ### Train  = 49%
    ### Valid  = 21%
    ### Test   = 30%
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.3, random_state= 73)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.3, random_state = 28)
    del X_tmp
    del y_tmp

    ### Select features for training
    #X_train_ready = X_train[features]
    #X_train_ready = X_train_ready.values
    #y_train_ready = y_train.values

    ### Select features for validation
    #X_val_ready = X_val[features]
    #X_val_ready = X_val_ready.values
    #y_val_ready = y_val.values

    ### Select features for testing
    #X_test_ready = X_test[features]
    #X_test_ready = X_test_ready.values
    #y_test_ready = y_test.values
    
    out_dict = {'X_train': X_train[features],
                'X_val': X_val[features],
                'X_test': X_test[features],
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test}
    
    return out_dict


def display_train_val_test_info(X_train_ready, y_train_ready,
                                X_val_ready, y_val_ready,
                                X_test_ready, y_test_ready):
    # Train val test percent
    train_per = len(y_train_ready) / (len(y_train_ready) + len(y_val_ready) + len(y_test_ready))
    val_per   = len(y_val_ready) / (len(y_train_ready) + len(y_val_ready) + len(y_test_ready))
    test_per  = len(y_test_ready) / (len(y_train_ready) + len(y_val_ready) + len(y_test_ready))
    print(f'train on {train_per*100:.2g}%  validate on {val_per*100:.2g}%   test on {test_per*100:.2g}%')
    print('')
    del train_per, val_per, test_per

    # data shapes
    print('X_train_ready shape:', X_train_ready.shape)
    print('y_train_ready shape:', y_train_ready.shape)
    print('X_val_ready shape:', X_val_ready.shape)
    print('y_val_ready shape:', y_val_ready.shape)
    print('X_test_ready shape:', X_test_ready.shape)
    print('y_test_ready shape:', y_test_ready.shape)
    print('')

    # make sure there are no nans anywhere
    print('NaNs in X_train_ready:', np.isnan(X_train_ready).sum())
    print('NaNs in y_train_ready:', np.isnan(y_train_ready).sum())
    print('NaNs in X_val_ready:', np.isnan(X_val_ready).sum())
    print('NaNs in y_val_ready:', np.isnan(y_val_ready).sum())
    print('NaNs in X_test_ready:', np.isnan(X_test_ready).sum())
    print('NaNs in y_test_ready:', np.isnan(y_test_ready).sum())
    print('')



# Do we need need a function? why not just define the model here?
## FFN model
def FFN_Model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, kernel_initializer='normal')])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    
    return model
    
    
#===============================================
# GLobal predictions
#================================================

def make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            ds_obs=None,
                            mask=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = tf_model.predict(df_all[features].values, verbose=0)
    df_all['spco2_error'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2_error']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_error = df_hpd['spco2_error'].values.reshape(dim_lat, dim_lon, dim_time)

    # spco2 for model
    spco2_model = ds_mod['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # spco2 hybrid predictions
    spco2_hpd = spco2_model - spco2_error

    # spco2 observations
    spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_socat':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'SOCATv2019 spco2'}),
        'spco2_model':(['lat', 'lon', 'time'], spco2_model, 
                       {'units': 'uatm',
                       'long_name':f'surface pressure of CO2 for {model_name} model'}),
        'spco2_error':(['lat', 'lon', 'time'], spco2_error,
                       {'units': 'uatm',
                       'long_name':f'learned errors [(spco2_error) = (spco2_SOCATv2019) - (spco2_{model_name})]'}),
        'spco2_hpd':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'corrected spco2 [(spco2_hpd) = (spco2_{model_name}) + (spco2_error)]'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'HPD-NN',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out




def make_global_predictions_xgb(ds_trim=None,
                            ds_mod=None,
                            ds_obs=None,
                            mask=None,
                            xgb_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = xgb_model.predict(df_all[features].values)
    df_all['spco2_error'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2_error']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_error = df_hpd['spco2_error'].values.reshape(dim_lat, dim_lon, dim_time)

    # spco2 for model
    spco2_model = ds_mod['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # spco2 hybrid predictions
    spco2_hpd = spco2_model - spco2_error

    # spco2 observations
    spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_socat':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'SOCATv2019 spco2'}),
        'spco2_model':(['lat', 'lon', 'time'], spco2_model, 
                       {'units': 'uatm',
                       'long_name':f'surface pressure of CO2 for {model_name} model'}),
        'spco2_error':(['lat', 'lon', 'time'], spco2_error,
                       {'units': 'uatm',
                       'long_name':f'learned errors [(spco2_error) = (spco2_SOCATv2019) - (spco2_{model_name})]'}),
        'spco2_hpd':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'corrected spco2 [(spco2_hpd) = (spco2_{model_name}) + (spco2_error)]'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'HPD-XGB',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out



def make_global_predictions_xgb_base(ds_trim=None,
                            ds_obs=None,
                            mask=None,
                            xgb_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = xgb_model.predict(df_all[features].values)
    df_all['spco2'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_hpd = df_hpd['spco2'].values.reshape(dim_lat, dim_lon, dim_time)

    # spco2 observations
    spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_socat':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'SOCATv2019 spco2'}),
        'spco2':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'XGB based spco2'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'XGB',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out



def make_global_predictions_nn_base(ds_trim=None,
                            ds_mod=None,
                            ds_obs=None,
                            mask=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = tf_model.predict(df_all[features].values, verbose=0)
    
    df_all['spco2'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_hpd = df_hpd['spco2'].values.reshape(dim_lat, dim_lon, dim_time)

    # spco2 observations
    spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_socat':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'SOCATv2019 spco2'}),
        'spco2':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'XGB based spco2'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'NN',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out

def make_global_predictions_xgb_osse(ds_trim=None,
                            ds_obs=None,
                            mask=None,
                            xgb_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = xgb_model.predict(df_all[features].values)
    df_all['spco2'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_hpd = df_hpd['spco2'].values.reshape(dim_lat, dim_lon, dim_time)

    #spco2_obs = df
    # spco2 observations
    spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_cesm':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'CESM spco2'}),
        'spco2':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'XGB based spco2'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'RFR - OSSE',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out




def make_global_predictions_osse_target(ds_trim=None,
                            ds_mod=None,
                            ds_obs=None,
                            mask=None,
                            xgb_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None):
    '''
    make_global_predictions(ds_trim=None,
                            ds_mod=None,
                            tf_model=None,
                            date_start=None,
                            date_end=None,
                            features=None, 
                            features_plus_coords=None, 
                            model_name=None)
    makes preductions on global map
    
    todo
    - two products : pco2 product and fgco2 product
    - [spco2_socat, networkmask, fgco, ice, aco2, kw, sol, seamask]
    '''
    # to make date in attributes
    from datetime import date
    
    # Latitude, longitude vectors
    lon = ds_trim['lon']
    lat = ds_trim['lat']
    time = ds_trim['time']

    # create a time,lat,lon skeleton
    skeleton_xr = ds_trim.to_dataframe().reset_index()
    skeleton_xr = skeleton_xr[['time','lat','lon']]

    # Define full dataframe to make predictions on
    df_all = ds_trim.to_dataframe().reset_index()
    df_all = df_all[features_plus_coords].dropna()

    # make predictions and put into dataframe
    y_pred = xgb_model.predict(df_all[features].values)
    df_all['spco2_error'] = y_pred

    # Merge predictions with df_all dataframe
    df_hpd = skeleton_xr.merge(df_all[['time', 'lat', 'lon', 'spco2_error']], 
                               how = 'left', on = ['time','lat','lon'])

    # Get dimensions to reshape data
    dim_time, dim_lat, dim_lon = ds_trim['spco2'].shape
    
    # spco2 error monthly
    spco2_error = df_hpd['spco2_error'].values.reshape(dim_lat, dim_lon, dim_time)

    # spco2 for model
    spco2_model = ds_mod['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # spco2 hybrid predictions
    spco2_hpd = spco2_model - spco2_error

    # spco2 observations
    #spco2_socat = ds_obs['spco2'].sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values
    spco2_socat = ds_obs['spco2'].compute().sel(time=slice(date_start, date_end)).transpose('lat','lon','time').values

    # Put into xarray dataset
    ds_out = xr.Dataset(
        {
        'spco2_cesm':(['lat', 'lon', 'time'], spco2_socat,
                       {'units': 'uatm',
                       'long_name':f'CESM spco2'}),
        'spco2_model':(['lat', 'lon', 'time'], spco2_model, 
                       {'units': 'uatm',
                       'long_name':f'surface pressure of CO2 for {model_name} model'}),
        'spco2_error':(['lat', 'lon', 'time'], spco2_error,
                       {'units': 'uatm',
                       'long_name':f'learned errors [(spco2_error) = (spco2_SOCATv2019) - (spco2_{model_name})]'}),
        'spco2_hpd':(['lat', 'lon', 'time'], spco2_hpd,
                       {'units': 'uatm',
                       'long_name':f'corrected spco2 [(spco2_hpd) = (spco2_{model_name}) + (spco2_error)]'}),  
        'mask':(['lat', 'lon'], mask.values,
                       {'units': '1=HPD domain',
                       'long_name':'prediction domain. removed marginal seas and 1 degree from coast.'}),
        },

        coords={
        'time': (['time'], time),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        },

        attrs={
        'institution':'LDEO, Palisades, New York',
        'institution_id':'LDEO',
        'model_id':'HPD-XGB',
        'run_id':'v2019',
        'contact':'Luke Gloege (gloege@ldeo.columbia.edu)',
        'creation_date':f'{date.today().strftime("%Y-%m-%d")}',
        })
    
    # Transpose output
    ds_out = ds_out.transpose('time','lat','lon')
    
    return ds_out