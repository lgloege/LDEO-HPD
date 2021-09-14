'''
calculates fgco2 from spco2

there needs to be an fgco2 directory already made, poorly coded
'''
import air_sea_co2_exchange as ase
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob

def main():
    # ----------------------------------------
    # directories
    # ----------------------------------------
    data_dir = '/home/gloege/projects/ldeo_hpd/data/model_output/XGB'
    era_dir = '/local/data/artemis/workspace/gloege/data/ERAinterim' # ERAinterim_1x1_monthly-mean-and-var_1982-2016.nc
    xco2_dir = '/local/data/artemis/workspace/gloege/data/globalview' # globalview_1x1_mon_197901-201901.nc
    input_dir = '/home/gloege/projects/ldeo_hpd/data/interim' 

    # ----------------------------------------
    # spco2 datasets
    # ----------------------------------------
    ds_spco2 = xr.merge([xr.open_dataset(fl) for fl in glob.glob(f'{data_dir}/*.nc')])

    # ----------------------------------------
    # variable names in ds_spco2
    # ----------------------------------------
    #variables = ['corrected_spco2_CCSM',
    #             'corrected_spco2_MPI',
    #             'corrected_spco2_NEMO',
    #             'corrected_spco2_NORESM',
    #             'corrected_spco2_RECOM']
    
    variables = ['corrected_cesm_spco2_1x1_A',
            'corrected_recom_jra_spco2_1x1_A',
            'corrected_mpi_spco2_1x1_A',
            'corrected_cnrm_spco2_1x1_A',       
            'corrected_noresm_spco2_1x1_A',    
            'corrected_planktom_spco2_1x1_A',
            'corrected_princeton_spco2_1x1_A',
            'corrected_csiro_spco2_1x1_A',
            'corrected_ipsl_spco2_1x1_A',]


    # ========================================
    # rename spco2 variable
    # ========================================
    ## Loop over pre
    for var in variables:
        print(var)
        # ----------------------------------------
        # rename spco2 variable
        # ----------------------------------------
        ds_spco2['spco2'] = ds_spco2[f'{var}']

        # ----------------------------------------
        # merge dataset togther
        # ----------------------------------------
        ds = xr.merge([
            xr.open_dataset('/local/data/artemis/observations/CO2_flux_inputs/processed/flux_variables.nc').\
                drop(['mld_clim','u10_mean','u10_std','xco2']).sel(time=slice('1982-01','2016-12')),
            xr.open_dataset(f'{era_dir}/ERAinterim_1x1_monthly-mean-and-var_1982-2016.nc').\
                sel(time=slice('1982-01','2016-12')),
            xr.open_dataset(f'{xco2_dir}/globalview_1x1_mon_197901-201901.nc').\
                sel(time=slice('1982-01','2016-12')),
            ds_spco2['spco2'].sel(time=slice('1982-01','2016-12')),])


        # ----------------------------------------
        # calculate fgco2
        # ----------------------------------------
        # get the model name
        model = var.split('corrected_')[1].split('_spco2_1x1_A')[0]

        # calculate fgco2
        ds_fgco2 = ase.calculate_fgco2(T=ds['sst'], 
                            S=ds['sss'], 
                            xCO2=ds['xco2'], 
                            P=ds['mslp'], 
                            pCO2_sw=ds['spco2'], 
                            u_mean=ds['u10_mean'], 
                            u_var=ds['u10_var'],  
                            iceFrac=ds['ifrac'],
                            scale_factor=0.27).to_dataset(name=f'fgco2_{model}')

        # ----------------------------------------
        # save netcdf file
        # ----------------------------------------
        out_dir='/home/gloege/projects/ldeo_hpd/data/model_output/XGB/fgco2'
        ds_fgco2.to_netcdf(f'{out_dir}/XGB_fgco2_{model}_mon_1x1_198201-201612.nc')

    return None

if __name__=='__main__':
    main()