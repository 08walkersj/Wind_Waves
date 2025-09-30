# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:23:37 2021

@author: admin
"""

import datetime as dt
import os
import read_wind_waves_file
import process_data
import plotting_spectrogram as pltsp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import akr_burst_search
import matplotlib as mpl
import matplotlib.pyplot as plt

def daily_raw_spectrogram(year,month,day,ax):
    # Define filename
    date=dt.datetime(year,month,day)
    fstart='wi_wa_rad1_l2_'
    fend='_v01.dat'
    filename=fstart+date.strftime('%Y%m%d')+fend

    # Define file directory
#    filedir='/home/a/arf18/dias_akr/james_python_example/wind_rad1_l2_data_doy227_232/'
    filedir=r'Users\admin\Documents\data\wind_raw'#str(year)

    #Stick them together
    #filepath = os.path.join(filedir,filename)
    filepath=os.path.join('C:'+os.sep,filedir,str(year),filename) #.replace("\","/")
    print('Reading Wind WAVES data from:',filepath)

    # not sure what this does yet! reading in the data I guess,
    #	but there are two variables on the LHS?!
    l2_obj, n_sw = read_wind_waves_file.read_l2_hres(filepath)
    
    l2_df = process_data.concat_l2_sweeps(l2_obj, n_sw)
    antenna = 'S'

    power_label = 'AMPL_' + antenna
    dtime_label = 'DATETIME_' + antenna
    # James: get 2d numpy array with 3 min spectra, and sorted frequency channels
    plt_arr, freqs, _ = pltsp.spectrogram_array(l2_df, power_label,
        'SWEEP', 'FREQ', dtime_label)

    # Defining some stuff for the plotting, in a form wanted by the dynamic_spectrogram code
    plot_kwargs = {'log_color': True,
        'color_map': 'viridis',
        'colorbar_label': ' '.join(['Received Power', r'($\mu V^{2}Hz^{-1}$)']),
        'title': True}

    # Define axis limits
    dt_bounds = pltsp.datetime_xaxis_limits(l2_df[dtime_label], '1H', False)	#tuple

    # Plot the spectrogram using the spectrogram code and axis limits and options defined above    
    ax = pltsp.dynamic_spectrogram(ax, plt_arr, freqs, *dt_bounds, **plot_kwargs)

    # Presumably sorting out labels and limits for the x-axis
    date_args = ('H', 'H', 1, 4)
    ax = pltsp.date_xaxis_setup(ax, l2_df[dtime_label],
        *date_args)
    ax.set_xlim(dt_bounds)
    return ax

def read_waves_raw_day(year,month,day):
    # Define filename
    date=dt.datetime(year,month,day)
    fstart='wi_wa_rad1_l2_'
    fend='_v01.dat'
    filename=fstart+date.strftime('%Y%m%d')+fend

    # Define file directory
#    filedir='/home/a/arf18/dias_akr/james_python_example/wind_rad1_l2_data_doy227_232/'
    filedir=r'Users\admin\Documents\data\wind_raw'#str(year)

    #Stick them together
    #filepath = os.path.join(filedir,filename)
    filepath=os.path.join('C:'+os.sep,filedir,str(year),filename) #.replace("\","/")
    print('Reading Wind WAVES data from:',filepath)

    # not sure what this does yet! reading in the data I guess,
    #	but there are two variables on the LHS?!
    l2_obj, n_sw = read_wind_waves_file.read_l2_hres(filepath)
    
    l2_df = process_data.concat_l2_sweeps(l2_obj, n_sw)
    antenna = 'S'
    
    return l2_df

def return_spectrogram_axis(l2_df,ax):
    # From James' code
    
    antenna = 'S'

    power_label = 'AMPL_' + antenna
    dtime_label = 'DATETIME_' + antenna
    # James: get 2d numpy array with 3 min spectra, and sorted frequency channels
    plt_arr, freqs, _ = pltsp.spectrogram_array(l2_df, power_label,
        'SWEEP', 'FREQ', dtime_label)

 #   print('plt_arr shape',plt_arr.shape)

    # Defining some stuff for the plotting, in a form wanted by the dynamic_spectrogram code
    plot_kwargs = {'log_color': True,
        'color_map': 'viridis',
        'colorbar_label': ' '.join(['Received Power', r'($\mu V^{2}Hz^{-1}$)']),
        'title': True}

    # Define axis limits
    dt_bounds = pltsp.datetime_xaxis_limits(l2_df[dtime_label], '1H', False)	#tuple
#    print(l2_df[dtime_label],l2_df[dtime_label].min(),l2_df[dtime_label].max())
 #   print('dt_bounds',dt_bounds,type(dt_bounds))
   # print(pd.date_range(*dt_bounds, periods=plt_arr.shape[1]+1))
    
   # print(pd.date_range(*dt_bounds, periods=plt_arr.shape[1]+1).shape)
   # print(l2_df.DATETIME_S.shape)
    
    #fig,ax=plt.subplots()
    #ax.plot(pd.date_range(*dt_bounds, periods=plt_arr.shape[1]+1) - l2_df.DATETIME_S)

    # Plot the spectrogram using the spectrogram code and axis limits and options defined above    
    ax = pltsp.dynamic_spectrogram(ax, plt_arr, freqs, *dt_bounds, **plot_kwargs)

    # Presumably sorting out labels and limits for the x-axis
    date_args = ('H', 'H', 1, 4)
    ax = pltsp.date_xaxis_setup(ax, l2_df[dtime_label],
        *date_args)
    ax.set_xlim(dt_bounds)
 
    return ax

def return_spec_axis(l2_df,ax):
    # Writing as in my other spectrogram codes
    
    # Take S antenna amplitudes, and take mean of all amps at same freq and sweep
    
    #l2_df=read_waves_raw_day(2002,8,9)
    
    # Find the unique sweep numbers and frequencies
    sweeps=l2_df.sweep_gen.unique()
    
    #print(sweeps)
    #return
    freqs=np.sort(l2_df.FREQ.unique())
    
    datetime_ut=[]
    intensity=[]
    freq=[]
    
    # Using Z antenna, because James said to
    # quote from slack:
    # be aware that with the current data on DIAS the S antenna (or S') are not used, 
    #   and only the Z received power (AMPL_Z) is calibrated. If you plot the S and Z
    #   data as spectrograms for a particularly active AKR period you may see extra 
    #   signal (usually intensifications for all frequencies during a sweep) in the 
    #   S antenna which is saturation of the electronics as the emission is so intense
    #   - more of a problem for the 10x longer, and more sensitive S antenna than the 
    #   shorter Z antenna
    for i in range(sweeps.size):
        #print(sweeps[i])
        for j in range(freqs.size):
            # Take average of the multiple intensity measurements in this SWEEP and at this FREQ
            intensity.append( np.nanmean( l2_df['AMPL_Z'].loc[ (l2_df.sweep_gen == sweeps[i]) & (l2_df.FREQ == freqs[j]) ] ) )
            # Select the minimum datetime in this SWEEP - start of the integration period
            datetime_ut.append( np.min( l2_df['DATETIME_Z'].loc[l2_df.sweep_gen == sweeps[i]] ) )
            freq.append(freqs[j])
            

    
    raw_df=pd.DataFrame({'datetime_ut':datetime_ut,'freq':freq, 'mask_flux_si':intensity}) 
    #print(raw_df)

    #fig,ax=plt.subplots()
    ax,x_arr,y_arr,z_arr=akr_burst_search.return_masked_akr_axis(raw_df,ax,no_cbar=True,flux_tag='mask_flux_si')
    
    im_m = ax.pcolormesh(x_arr, y_arr, z_arr,cmap='viridis',
                            norm=mpl.colors.LogNorm())
    ax.set_yscale('log')
    
    # NOTE calling colorbar with plt and not fig, and no explicit ax kwarg
    # puts colorbar on first ax object it finds, whether global or not etc 
    cbar = plt.colorbar(im_m, ax=ax, label=r'Received Power ($\mu V^{2}Hz^{-1}$)')
    cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
    cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    
    return ax,x_arr,y_arr,z_arr