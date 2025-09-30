#! /usr/bin/env python
# -*- coding: latin-1 -*-

"""
Python module to read a Wind/Waves data file.
@author: X.Bonnin (LESIA)

Modified by J.Waters
"""

import struct

__author__ = "Xavier Bonnin"
__date__ = "03-MAR-2013"
__version__ = "1.00"

__all__ = ["read_l2_hres"]


class Waves_data:
    def __init__(self,header,data):
        self.header = header
        self.data = data

def read_l2_hres(filepath, verbose=True):

    """
    Method to read a Wind/Waves l2 high resolution data file"
    """

    header_fields = ("P_FIELD","JULIAN_DAY_B1","JULIAN_DAY_B2","JULIAN_DAY_B3","MSEC_OF_DAY",
                     "RECEIVER_CODE","JULIAN_SEC_FRAC","YEAR","MONTH","DAY",
                     "HOUR","MINUTE","SECOND","JULIAN_SEC_FRAC",
                     "ISWEEP","IUNIT","NPBS","SUN_ANGLE","SPIN_RATE","KSPIN","MODE","LISTFR","NFREQ",
                     "ICAL","IANTEN","IPOLA","IDIPXY","SDURCY","SDURPA",
        "NPALCY","NFRPAL","NPALIF","NSPALF","NZPALF")
    header_dtype = '>bbbbihLhhhhhhfihhffhhhhhhhhffhhhhh'

    header = []; data = []; nsweep = 1
    with open(filepath,'rb') as frb:
        while (True):
            try:
                if verbose:
                    print("Reading sweep #{}".format(nsweep))
                # Reading number of octets in the current sweep
                block = frb.read(4)
                if (len(block) == 0): break
                loctets1 = struct.unpack('>i', block)[0]
                # Reading header parameters in the current sweep
                block = frb.read(80)
                header_i = dict(zip(header_fields, struct.unpack(header_dtype, block)))
                npalf = header_i['NPALIF']; nspal = header_i['NSPALF']; nzpal = header_i['NZPALF']
                # Reading frequency list (kHz) in the current sweep
                block = frb.read(4 * npalf)
                freq = struct.unpack('>' + 'f' * npalf,
                                     block)
                # Reading intensity and time values for S/SP in the current sweep
                block = frb.read(4 * npalf * nspal)
                Vspal = struct.unpack('>' + 'f' * npalf * nspal,
                                      block)
                block = frb.read(4 * npalf * nspal)
                Tspal = struct.unpack('>' + 'f' * npalf * nspal,
                                      block)
                # Reading intensity and time values for Z in the current sweep
                block = frb.read(4 * npalf * nzpal)
                Vzpal = struct.unpack('>' + 'f' * npalf * nzpal,
                                      block)
                block = frb.read(4 * npalf * nzpal)
                Tzpal = struct.unpack('>' + 'f' * npalf * nzpal,
                                      block)
                # Reading number of octets in the current sweep
                block = frb.read(4)
                loctets2 = struct.unpack('>i', block)[0]
                if (loctets2 != loctets1):
                    print("Error reading file!")
                    return None
                breakpoint()
            except EOFError:
                print("End of file reached")
                break
            else:
                header.append(header_i)
                data.append({"FREQ": freq,
                             "VSPAL": Vspal,
                             "VZPAL": Vzpal,
                             "TSPAL": Tspal,
                             "TZPAL": Tzpal})
                print(Tzpal)
                nsweep+=1
        # print('All sweeps read')

    return Waves_data(header, data), nsweep-1
#!usr/bin/env python3

import numpy as np
import pandas as pd


def get_sweep_datetime(waves_l2, sweep_i):
    """
    Given a header/data block of wind/waves L2 data,
    extract the datetime object corresponding to the start of the sweep cycle
    given by `sweep_i`

    Parameters
    ----------
    waves_l2: Waves_data object
        see read_wind_waves_file.read_l2_hres() output

    sweep_i: int
        index of sweep cycle of interest
        0 <= sweep_i <= max_sweep-1

    Returns
    -------
    date_time: datetime object
        the datetime of the beginning of the sweep cycle (cycle # sweep_i+1)
    """
    
    year = waves_l2.header[sweep_i]['YEAR']
    month = waves_l2.header[sweep_i]['MONTH']
    day = waves_l2.header[sweep_i]['DAY']
    
    date = pd.Timestamp(year, month, day)

    msec = waves_l2.header[sweep_i]['MSEC_OF_DAY']

    datetime = date + pd.Timedelta(msec, 'ms')	#changed from 'milli' to 'ms to fit with the pandas version I have on Spectre, James found this out online 05/10/2020

    return datetime


def check_sweep_cycle(sweep_cycle_dict):
    """
    Ensure that appropriate tags are present within the sweep cycle header
    file - e.g polarisation present, optimal antennae used, correct units etc
    
    Parameters
    ----------
    sweep_cycle_dict : dict
        contains relevant header parameters and amplitude/time data for each
        channel
        
    Returns
    -------
    None
    
    
    NB - may require changes, asserts not necessary for less stringent
    requirements eg eq_dipole
    
    """
    # Check receiver amplitudes in microV^2 Hz^-1
    assert sweep_cycle_dict['i_units'] == 3, \
        "Receiver amplitude are not in the correct units."

    # Check solar top quality
    assert sweep_cycle_dict['k_spin'] == 0, \
        "Poor solar top quality - interpolation has been applied. (?)"
    
    # Check data acquired using measure/list mode
    assert sweep_cycle_dict['data_acq_mode'] == 3, \
        "Amplitudes compiled using unexpected data acquisition mode"
        
    # Check S and Z antenna used in SUM mode
    assert sweep_cycle_dict['ant_config'] == 2, \
        "S and Z antennae not configured in SUM mode"
        
    # Check polarisation present
    assert sweep_cycle_dict['pol_present'] == 1, \
        "Polarisation not present for this sweep cycle."
    
    # Check appropriate, longer equatorial antenna used
    assert sweep_cycle_dict['eq_dipole'] == 1, \
        "Shorter equatorial plane Y-antenna used"
        
    # Check same number of samples per measurement step observed for S and SP
    # as for Z
    assert sweep_cycle_dict['n_s_samples_per_step'] == \
           2 * sweep_cycle_dict['n_z_samples_per_step'], \
           "Incompatible number of measurements for each antennae"
    
    return None


def convert_sweep_cycle_block(waves_l2, sweep_i):
    """
    Transfer data from dictionary into pandas dataframe for one 
    header/data sweep cycle block for all antennae
    
    List of antennae channels used:

        Z is spin-axis-aligned antenna
        S is synthetic antenna (Wind/Waves SUM mode)
        SP is synthetic antenna with phase shift
    
    Dictionary keys are:
        ('VZPAL/TZPAL' for Z, 'VSPAL/TSPAL' for S/Sp)

    Parameters
    ----------
    waves_l2: Waves_data class object

    sweep_i: int
        Indexer used to access appropriate header/data block of waves_l2
        sweep # - 1


    Returns
    -------
    sweep_out : dict

    """

    # Extract frequency bins (PalkHz)
    freqs = np.array(waves_l2.data[sweep_i]['FREQ'])
    
    # Checking number of measurement steps correspond to recorded value
    #(Could move into test_ function - tests all these values between header/data)
    if np.shape(freqs)[0] != waves_l2.header[sweep_i]['NPALIF']:
        
        raise AssertionError('Line 150 assertion raised (make_wav_l3_rad1.py)')

    # Extract measurement times and amplitudes for each antenna
    # S
    times_s = np.array(waves_l2.data[sweep_i]['TSPAL'])[::2]
    ampl_s = np.array(waves_l2.data[sweep_i]['VSPAL'])[::2]
    N_measurements_s = waves_l2.header[sweep_i]['NSPALF'] / 2
    
    # S_prime
    times_sp = np.array(waves_l2.data[sweep_i]['TSPAL'])[1::2]
    ampl_sp = np.array(waves_l2.data[sweep_i]['VSPAL'])[1::2]
    N_measurements_sp = waves_l2.header[sweep_i]['NSPALF'] / 2
    
    # Z
    times_z = np.array(waves_l2.data[sweep_i]['TZPAL'])
    ampl_z = np.array(waves_l2.data[sweep_i]['VZPAL'])
    N_measurements_z = waves_l2.header[sweep_i]['NZPALF']
    
    assert (N_measurements_s == N_measurements_sp) and \
           (N_measurements_s == N_measurements_z) and \
           (N_measurements_sp == N_measurements_z), \
           "Unequal number of measurements made between each antenna."
           
    N_samples = N_measurements_s

    freqs = np.repeat(freqs, N_samples)
    freqs = np.array([int(f) for f in freqs])
    
    # Pull sun angle (see DSUNCY in docs) and spin rate (see DSSPIN in docs)
    sun_angle = waves_l2.header[sweep_i]['SUN_ANGLE']
    spin_rate = waves_l2.header[sweep_i]['SPIN_RATE']
    # 1-indexed sweep cycle number
    i_sweep = waves_l2.header[sweep_i]['ISWEEP']
    # L2 intensity units tag
    i_unit = waves_l2.header[sweep_i]['IUNIT']
    # solar top quality (?) tag
    k_spin = waves_l2.header[sweep_i]['KSPIN']
    # data acquisition mode
    mode = waves_l2.header[sweep_i]['MODE']
    # antenna configuration tag
    i_ant = waves_l2.header[sweep_i]['IANTEN']
    # polarisation flag
    i_pol = waves_l2.header[sweep_i]['IPOLA']
    # eq dipole used
    i_dipole_xy = waves_l2.header[sweep_i]['IDIPXY']
    # sweep cycles duration (seconds)
    sweep_dur = waves_l2.header[sweep_i]['SDURCY']
    # measurement step duration (duration to sample one frequency)
    step_dur = waves_l2.header[sweep_i]['SDURPA']
    
    # number of measurement steps (sets of 8 frequency samples)
    n_meas_steps = waves_l2.header[sweep_i]['NPALCY']
    # number of frequencies in measurement step (assumed 1)
    n_freq_per_step = waves_l2.header[sweep_i]['NFRPAL']
    # number of total measurement steps 
    n_meas_step_f = waves_l2.header[sweep_i]['NPALIF']
    # number of unique frequencies measured
    n_freq = waves_l2.header[sweep_i]['NFREQ']
    
    # get number of measurement steps for S/SP and Z antennae
    n_samp_per_step_S = waves_l2.header[sweep_i]['NSPALF']
    n_samp_per_step_Z = waves_l2.header[sweep_i]['NZPALF']    
    
    swcy_tag = 0
    try:
        assert n_freq_per_step == 1, \
            "More than one frequency sampled per measurement step"
    except AssertionError:
        swcy_tag = 1
    
    # Still have datetime of start of sweep cycle
    sweep_start_date = get_sweep_datetime(waves_l2, sweep_i)

    # unpacking datetimes from time arrays
    datetime_s, datetime_sp, datetime_z = [np.array([sweep_start_date + 
        pd.Timedelta(t, 's') for t in times]) for times in [times_s, times_sp, times_z]]
    
    block = pd.DataFrame({'FREQ': freqs,
                          'TIME_S': times_s,
                          'TIME_SP': times_sp,
                          'TIME_Z': times_z,
                          'AMPL_S': ampl_s,
                          'AMPL_SP': ampl_sp,
                          'AMPL_Z': ampl_z,
                          'DATETIME_S': datetime_s,
                          'DATETIME_SP': datetime_sp,
                          'DATETIME_Z': datetime_z})

    block['SWEEP'] = block.shape[0] * [sweep_i]
    
    sweep_out = {"sweep_tag": swcy_tag,
                 "i_sweep": i_sweep,
                 "i_units": i_unit,
                 "k_spin": k_spin,
                 "data_acq_mode": mode,
                 "ant_config": i_ant,
                 "pol_present": i_pol,
                 "eq_dipole": i_dipole_xy,
                 "sweep_dur": sweep_dur,
                 "step_dur": step_dur,
                 "start_date": sweep_start_date,
                 "sun_angle": sun_angle,
                 "spin_rate": spin_rate,
                 "n_s_samples_per_step": n_samp_per_step_S,
                 "n_z_samples_per_step": n_samp_per_step_Z,
                 "n_steps": n_meas_steps,
                 "n_freqs": n_freq,
                 "data": block}
    
    # Apply tag if sweep cycle differs from base expectation
    try:
        check_sweep_cycle(sweep_out)                          
    except AssertionError:
        sweep_out['sweep_tag'] = 1
        
    return sweep_out


def concat_l2_sweeps(l2_object, n_sw):
    """
    Convert sweep cycle data object into dataframe for the whole day
    
    NB will have to convert times to datetimes prior to concatenation, 
    as times are stored as seconds from sweep cycle start
    """
    dfs = np.empty(n_sw, dtype=object)

    for i in range(n_sw):
        
        sweep_dict = convert_sweep_cycle_block(l2_object, i)
        
        sweep_df = sweep_dict['data']

        sweep_df['sweep_start_date'] = np.repeat(sweep_dict['start_date'],
            sweep_df.shape[0])

        sweep_df['sun_angle'] = np.repeat(sweep_dict['sun_angle'],
            sweep_df.shape[0])

        sweep_df['spin_rate'] = np.repeat(sweep_dict['spin_rate'],
            sweep_df.shape[0])

        sweep_df['SWEEP'] = np.repeat(sweep_dict['i_sweep'],
            sweep_df.shape[0])

        sweep_df['sweep_flag'] = np.repeat(sweep_dict['sweep_tag'],
        sweep_df.shape[0])

        if sweep_dict['sweep_tag'] == 0:

            dfs[i] = sweep_df    
    
        else:

            dfs[i] = None
        
    day_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    return day_df

if (__name__=="__main__"):
    from wind_raw_reading_code.process_data import concat_l2_sweeps
    import pandas as pd
    print("Python module to read Wind/Waves data file.")
    file= '/home/simon/Documents/WIND_Data/raw/1999/wi_wa_rad1_l2_19990118_v01.dat'
    data, sweep= read_l2_hres(file)
    # data2=concat_l2_sweeps(data, sweep)
    # file= '/home/simon/Documents/WIND_Data/raw/1999/wi_wa_rad1_l2_19990117_v01.dat'
    # data, sweep= read_l2_hres(file)
    # data3=concat_l2_sweeps(data, sweep)
    # data4= pd.concat([data2, data3])
    # data4.sort_values('DATETIME_S', inplace=True)

