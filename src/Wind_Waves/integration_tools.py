#%% Imports
import pandas as pd
import numpy as np

#%% Definitions

def fit_pandas(df, sweep, frequency, flux):
    """
    Computes linear segments by calculating slopes and intercepts between consecutive frequency-flux pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sweep, frequency, and flux data.
    sweep : str
        Column name representing sweep identifiers.
    frequency : str
        Column name representing frequency values.
    flux : str
        Column name representing flux values.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated slopes, intercepts, and frequency segment boundaries.

    Example
    -------
    >>> df = pd.DataFrame({"sweep": [1, 1, 1], "frequency": [10, 20, 30], "flux": [5, 10, 15]})
    >>> fit_pandas(df, "sweep", "frequency", "flux")
    """
    freqs, flux = df[[frequency, flux]].values.T  # Extract frequency and flux as separate arrays
    slopes = np.diff(flux) / np.diff(freqs)  # Compute slopes between consecutive points
    intercepts = flux[:-1] - slopes * freqs[:-1]  # Compute intercepts based on slope and frequency
    
    return pd.DataFrame({
        sweep: df[sweep].iloc[0],  # Assign the sweep value to each row
        'intercept': intercepts,  # Store computed intercept values
        'slope': slopes,  # Store computed slope values
        'f1': freqs[:-1],  # Store the first frequency of each segment
        'f2': freqs[1:]  # Store the second frequency of each segment
    })

def linear_segments(df, time='datetime_ut', frequency='FREQ', sweep='SWEEP', flux='AMPL_Z', 
                    preserve_cols=['sweep_start_date'], preserve_funcs=['min', 'max', 'median'],
                    preserve_col_suffix=[]):
    """
    Processes a DataFrame to compute linear segments based on frequency and flux values, ensuring sweep uniqueness.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with sweep, frequency, and flux columns.
    time : str, optional
        Column name representing time values. Default is 'Date_UTC'.
    frequency : str, optional
        Column name representing frequency values. Default is 'FREQ'.
    sweep : str, optional
        Column name representing sweep identifiers. Default is 'SWEEP'.
    flux : str, optional
        Column name representing flux values. Default is 'AMPL_Z'.
    preserve_cols : list, optional
        Columns to preserve in the final DataFrame. Default is ['DATETIME_Z', 'sweep_start_date'].
    preserve_funcs : list, optional
        Aggregation functions to apply to preserved columns. Default is ['min', 'max', 'median'].
    preserve_col_suffix : list, optional
        Suffixes for preserved column names. Default is an empty list.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with computed linear segments and/or preserved columns.
    
    Example
    -------
    >>> linear_segments(data, preserve_cols=['Date_UTC'], time='Date_UTC', frequency='freq', sweep='SWEEP', flux='akr_flux_si_1au')
    """
    # Check if the first sweep spans more than 10 minutes
    if (df.loc[df[sweep] == df[sweep].min(), [time]].max() -
        df.loc[df[sweep] == df[sweep].min(), [time]].min()).values[0] > np.timedelta64(10, 'm'):
        raise ValueError('Sweep number duplication covering more than 10 minutes')
    
    # Group by sweep and frequency, computing mean flux
    df_sweep_resampled = df.groupby([sweep, frequency])[flux].mean().reset_index()
    
    # Apply the fit_pandas function to compute linear segments
    fit_results = df_sweep_resampled.groupby(sweep, group_keys=False)[[sweep, frequency, flux]]\
        .apply(fit_pandas, sweep=sweep, frequency=frequency, flux=flux).reset_index(drop=True)
    
    if len(preserve_cols):  # If there are columns to preserve
        if not len(preserve_funcs):  # Ensure preservation functions are provided
            raise ValueError(f'No preservation functions. To preserve columns: {preserve_cols}, '
                             'functions are needed to describe how to preserve the columns')
        
        if not len(preserve_col_suffix):  # Assign suffixes based on function names if not provided
            preserve_col_suffix = [str(func) for func in preserve_funcs]
        
        g = df.groupby([sweep])[preserve_cols]  # Group by sweep for aggregation
        preserved = g.agg(preserve_funcs)  # Apply aggregation functions
        preserved.columns = [f"{col}_{stat}" for col, stat in preserved.columns]  # Rename columns
        preserved.reset_index(drop=False, inplace=True)  # Reset index to merge
        
        return fit_results.merge(preserved, on=sweep)  # Merge results with preserved data
    return fit_results

def linear_in_chunks(hdf5_file, output_file, chunk_size=100, in_key='main', out_key='linear_fit', **linear_segments_kwargs):
    """
    Loads and preforms the linear fit to HDF5 data in chunks on each sweep.
    The processed chunks are appended to an output HDF5 file.

    Parameters
    ----------
    hdf5_file : str
        Path to input HDF5 file.
    output_file : str
        Path to output HDF5 file.
    chunk_size : int, optional
        Number of sweeps to process in each chunk. Default is 100.

    Returns
    -------
    string
        path to output file.

    Example
    -------
    >>> process_in_chunks('input.h5', 'output.h5', chunk_size=50)
    """
    # Imports progressbar if available if not creates a dummy function
    try:
        from progressbar import progressbar
    except ImportError:
        def progressbar(*args, **kwargs):
            return args[0]
    if not 'Date_UTC' in linear_segments_kwargs:
        linear_segments_kwargs['preserve_cols']=['Date_UTC']
    if not 'sweep' in linear_segments_kwargs:
        linear_segments_kwargs['sweep']= 'SWEEP'
    # Open HDF5 file and retrieve unique sweep values
    with pd.HDFStore(hdf5_file, mode='r') as store:
        unique_sweeps = pd.Index(store.select_column(in_key, f'{linear_segments_kwargs["sweep"]}')).unique()

    # Loop through sweeps in chunks
    for i in progressbar(range(0, len(unique_sweeps), chunk_size), prefix='Looping SWEEP chunks: '):
        x1 = unique_sweeps[i]  # Start of chunk
        x2 = unique_sweeps[min(i + chunk_size - 1, len(unique_sweeps) - 1)]  # End of chunk
        
        # Load the relevant sweep data for the chunk
        chunk_data = pd.read_hdf(hdf5_file,
                                 key='main',
                                 where=f'{linear_segments_kwargs["sweep"]} >= {x1} & {linear_segments_kwargs["sweep"]} <= {x2}')
        
        # Process the chunk using linear segment fitting
        chunk_result = linear_segments(chunk_data, **linear_segments_kwargs)
        
        # Append the processed chunk to the output HDF5 file
        chunk_result.to_hdf(output_file, key=out_key, format='t', append=True, mode='a', data_columns=True)
    
    return output_file

def integrate_in_chunks(lin_fit_hdf5, output_file, flimits, chunk_size=100, in_key='linear_fit', out_key='integrated_power', **integrate_kwargs):
    """
    Integrate linear segment data in chunks from an HDF5 file and append results to an output HDF5 file.

    Parameters
    ----------
    lin_fit_hdf5 : str
        Path to the input HDF5 file containing linear fit data.
    output_file : str
        Path to the output HDF5 file where integrated results will be stored.
    flimits : tuple or pd.DataFrame
        Frequency limits for integration. Can be a tuple (fmin, fmax) for fixed limits or a DataFrame for variable limits.
    chunk_size : int, optional
        Number of sweeps to process in each chunk. Default is 100.
    in_key : str, optional
        Key under which to read the linear fit data in the input HDF5 file. Default is 'processed'.
    out_key : str, optional
        Key under which to store the integrated results in the output HDF5 file. Default is 'integrated_power'.
    **integrate_kwargs :
        Additional keyword arguments to pass to the `integrate` function.

    Returns
    -------
    output_file : str
        Path to the output HDF5 file containing the integrated results.

    Example
    -------
    >>> integrate_in_chunks('input_lin_fit.h5', 'output_integrated.h5', flimits=(40, 1040), chunk_size=200)
    'output_integrated.h5'
    """
    # Try to import progressbar for progress indication; if unavailable, use a dummy function
    try:
        from progressbar import progressbar
    except ImportError:
        def progressbar(*args, **kwargs):
            return args[0]

    # Open the HDF5 file and retrieve unique sweep values
    with pd.HDFStore(lin_fit_hdf5, mode='r') as store:
        unique_sweeps = pd.read_hdf(store, key=in_key, columns=['SWEEP']).SWEEP.unique()

    # Loop through sweeps in chunks for memory efficiency
    for i in progressbar(range(0, len(unique_sweeps), chunk_size), prefix='Looping SWEEP chunks: '):
        x1 = unique_sweeps[i]  # Start of chunk
        x2 = unique_sweeps[min(i + chunk_size - 1, len(unique_sweeps) - 1)]  # End of chunk

        # Load the relevant sweep data for the chunk
        chunk_data = pd.read_hdf(lin_fit_hdf5, key=in_key, where=f'SWEEP >= {x1} & SWEEP <= {x2}')

        # Integrate the chunk using the provided frequency limits and additional kwargs
        chunk_result = integrate(chunk_data, flimits, **integrate_kwargs)

        # Append the processed chunk to the output HDF5 file under the specified key
        chunk_result.to_hdf(output_file, key=out_key, format='t', append=True, mode='a', data_columns=True)

    return output_file
def integrate_linear_segment(row, f_min=40, f_max=1040):
    """
    Computes the definite integral of a linear segment within specified frequency limits.

    Parameters
    ----------
    row : pd.Series
        A row containing the segment's slope, intercept, and frequency boundaries.
    f_min : float, optional
        The lower limit of integration. Default is 40.
    f_max : float, optional
        The upper limit of integration. Default is 1040.

    Returns
    -------
    float
        The computed integral value for the segment.

    Example
    -------
    >>> row = pd.Series({'slope': 2, 'intercept': 1, 'f1': 50, 'f2': 100})
    >>> integrate_linear_segment(row)
    6375.0
    """
    m = row['slope']  # Extract the slope of the segment
    c = row['intercept']  # Extract the intercept
    f1, f2 = row['f1'], row['f2']  # Extract segment frequency boundaries
    
    f_int_min = max(f1, f_min)  # Determine the lower bound for integration
    f_int_max = min(f2, f_max)  # Determine the upper bound for integration
    
    if f_int_min >= f_int_max:  # Check if the segment is outside the integration range
        return 0
    
    # Compute the definite integral of the linear function over the segment
    integral = (m / 2) * (f_int_max**2 - f_int_min**2) + c * (f_int_max - f_int_min)
    return integral

def integrate_with_variable_limits(row, f_min, f_max):
    """
    Computes the definite integral of a linear segment with variable frequency limits.

    Parameters
    ----------
    row : pd.Series
        A row containing the segment's slope, intercept, and frequency boundaries.
    f_min : dict
        A dictionary mapping sweep identifiers to minimum frequency values.
    f_max : dict
        A dictionary mapping sweep identifiers to maximum frequency values.

    Returns
    -------
    float
        The computed integral value for the segment.

    Example
    -------
    >>> row = pd.Series({'slope': 2, 'intercept': 1, 'f1': 50, 'f2': 100, 'SWEEP': 1})
    >>> f_min = {1: 40}
    >>> f_max = {1: 90}
    >>> integrate_with_variable_limits(row, f_min, f_max)
    4450.0
    """
    m = row['slope']  # Extract the slope
    c = row['intercept']  # Extract the intercept
    f1, f2 = row['f1'], row['f2']  # Extract segment frequency boundaries
    sweep = row['SWEEP']  # Extract the sweep identifier
    
    f_int_min = np.max([f1, f_min.get(sweep, np.nan)])  # Get the lower integration limit
    f_int_max = np.min([f2, f_max.get(sweep, np.nan)])  # Get the upper integration limit
    
    if np.isnan(f_int_min):  # Handle missing frequency limits
        return np.nan
    if f_int_min >= f_int_max:  # Check if the segment is outside the integration range
        return 0
    
    # Compute the definite integral
    integral = (m / 2) * (f_int_max**2 - f_int_min**2) + c * (f_int_max - f_int_min)
    return integral

def sum_ints(x, dist=1.496e11):
    """
    Computes the sum of a series, ignoring NaN values.

    Parameters
    ----------
    x : pd.Series
        The series to sum.

    Returns
    -------
    float
        The sum of non-NaN values.

    Example
    -------
    >>> sum_ints(pd.Series([1, 2, np.nan, 4]))
    7.0
    """
    return np.nansum(x.values)*(dist**2)*1e3

def integrate(df, flimits, sweep='SWEEP', fmin='fmin', fmax='fmax', distance=1.496e11):
    """
    Computes the total integral per sweep, using either fixed or variable limits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the segment data.
    flimits : tuple or pd.DataFrame
        If a tuple, contains (fmin, fmax) fixed limits. If a DataFrame, provides per-sweep limits.
    sweep : str, optional
        Column name representing sweep identifiers. Default is 'SWEEP'.
    fmin : str, optional
        Column name representing the lower frequency limit. Default is 'fmin'. Not required if flimits is tuple.
    fmax : str, optional
        Column name representing the upper frequency limit. Default is 'fmax'. Not required if flimits is tuple.

    Returns
    -------
    pd.DataFrame
        DataFrame with integrated values per sweep.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'SWEEP': [1, 1, 2, 2],
    ...     'slope': [2, 2, 3, 3],
    ...     'intercept': [1, 1, 0, 0],
    ...     'f1': [50, 100, 200, 300],
    ...     'f2': [100, 150, 250, 350]
    ... })
    >>> flimits = (40, 140)
    >>> integrate(df, flimits)
       SWEEP  integral_40_140
    0      1           14850.0
    1      2           24000.0
    """
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    
    if isinstance(flimits, pd.DataFrame):  # Check if flimits is a DataFrame
        fmin = dict(zip(flimits[sweep], flimits[fmin]))  # Convert fmin column to dictionary
        fmax = dict(zip(flimits[sweep], flimits[fmax]))  # Convert fmax column to dictionary
        df['integral_variable_limits'] = df.apply(integrate_with_variable_limits, axis=1, f_min=fmin, f_max=fmax)
        return df.groupby(sweep)['integral_variable_limits'].apply(lambda x: sum_ints(x, distance)).reset_index()
    else:
        fmin, fmax = flimits  # Extract fixed frequency limits
        df[f'integral_{fmin}_{fmax}'] = df.apply(integrate_linear_segment, axis=1, f_min=fmin, f_max=fmax)
        return df.groupby(sweep)[f'integral_{fmin}_{fmax}'].apply(lambda x: sum_ints(x, distance)).reset_index()
def create_sweeps(data, time='datetime_ut', sweep='SWEEP', inplace=True):
    """
    Assigns unique sweep numbers to data based on time factorization.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing time values.
    time : str, optional
        Column name representing time values. Default is 'datetime_ut'.
    sweep : str, optional
        Column name to store the generated sweep identifiers. Default is 'SWEEP'.
    inplace : bool, optional
        If True, modifies the input DataFrame in place. If False, returns a modified copy. Default is True.

    Returns
    -------
    pd.DataFrame or None
        If inplace is False, returns a modified DataFrame with the assigned sweep numbers.
        If inplace is True, modifies the DataFrame directly and returns None.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'datetime_ut': ['2024-03-01 00:00:00', '2024-03-01 00:01:00', '2024-03-01 00:01:00',
    ...                     '2024-03-01 00:02:00', '2024-03-01 00:03:00', '2024-03-01 00:03:00']
    ... })
    >>> df['datetime_ut'] = pd.to_datetime(df['datetime_ut'])
    >>> create_sweeps(df)
    >>> print(df)
             datetime_ut  SWEEP
    0 2024-03-01 00:00:00      1
    1 2024-03-01 00:01:00      2
    2 2024-03-01 00:01:00      2
    3 2024-03-01 00:02:00      3
    4 2024-03-01 00:03:00      4
    5 2024-03-01 00:03:00      4
    """
    if not inplace:  # If inplace is False, create a copy to avoid modifying the original DataFrame
        data = data.copy()
    data.sort_values(time, inplace=True)
    
    data[sweep] = data[time].factorize()[0] + 1  # Assign unique sweep numbers based on factorized time values
    
    if not inplace:  # If inplace is False, return the modified DataFrame
        return data
def create_spins(data, time='DATETIME_Z', freq='FREQ', spin='spin', inplace=True):
    """
    Create a 'spin' column in a DataFrame by grouping consecutive rows 
    where the 'freq' column remains constant. 
    Each change in 'freq' increments the spin group number.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing at least a 'freq' column and a time column.
    time : str, default 'DATETIME_Z'
        The column name representing timestamps or chronological order for sorting.
    spin : str, default 'spin'
        The name of the new column to store spin group identifiers.
    inplace : bool, default True
        If True, modifies the DataFrame in place. 
        If False, returns a new DataFrame with the spin column added.

    Returns
    -------
    pandas.DataFrame or None
        Returns a new DataFrame if inplace=False, otherwise modifies in place and returns None.

    Notes
    -----
    - The function sorts the DataFrame by the `time` column before assigning spins.
    - The spin column is assigned based on cumulative changes in the `freq` column.
    """

    if not inplace:
        # Work on a copy if not modifying in place
        data = data.copy()

    # Ensure data is sorted by the time column
    data.sort_values(time, inplace=True)

    # Compare 'freq' values with previous row, cast to bool (True = change detected),
    # then take cumulative sum to assign a new group number for each change
    data[spin] = data[freq].diff().astype(bool).cumsum()

    # Return the modified copy if inplace=False
    if not inplace:
        return data

def create_burst_numbers(data, burst_list, time='datetime_ut', burst_number='Burst_Number', inplace=True):
    """
    Assigns burst numbers to data based on provided burst time intervals.

    Each data row is assigned a burst number if its timestamp falls within the time range of a corresponding burst
    event defined in `burst_list`. If the timestamp does not fall into any burst interval, it is assigned a default value of -1.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the time column.
    burst_list : pd.DataFrame
        DataFrame with burst events. Must include 'stime' (start time) and 'etime' (end time) columns.
    time : str, optional
        Column in `data` containing time values. Default is 'datetime_ut'.
    burst_number : str, optional
        Column name to assign the burst number. Default is 'Burst_Number'.
    inplace : bool, optional
        If True, modifies the input DataFrame in place. If False, returns a modified copy. Default is True.

    Returns
    -------
    pd.DataFrame or None
        If inplace is False, returns a modified DataFrame with assigned burst numbers.
        If inplace is True, modifies `data` in place and returns None.

    Example
    -------
    >>> data = pd.DataFrame({'datetime_ut': pd.date_range('2024-03-01', periods=5, freq='min')})
    >>> bursts = pd.DataFrame({
    ...     'stime': [pd.Timestamp('2024-03-01 00:00:00')],
    ...     'etime': [pd.Timestamp('2024-03-01 00:02:00')]
    ... })
    >>> create_burst_numbers(data, bursts)
    >>> print(data)
             datetime_ut  Burst_Number
    0 2024-03-01 00:00:00             1
    1 2024-03-01 00:01:00             1
    2 2024-03-01 00:02:00             1
    3 2024-03-01 00:03:00            -1
    4 2024-03-01 00:04:00            -1
    """
    # Ensure burst events are sorted by start time
    burst_list.sort_values('stime', inplace=True)

    # Assign unique burst numbers starting from 1
    burst_list[burst_number] = np.arange(len(burst_list)).astype(int) + 1

    if not inplace:
        data = data.copy()  # Create a copy if not modifying in place

    data.sort_values(time, inplace=True)  # Sort data chronologically

    # Helper function to assign burst number to each timestamp
    def assign_burst(t):
        row = burst_list[(burst_list['stime'] <= t) & (burst_list['etime'] >= t)]
        if not row.empty:
            return row[burst_number].values[0]
        return -1  # If no matching burst is found

    # Apply burst assignment to each row in the data
    data[burst_number] = data[time].apply(assign_burst).values

    if not inplace:
        return data  # Return the modified DataFrame if inplace=False

def get_akr_flims(burst_list='./fogg_akr_burst_list_1995_2004.csv',
                                         ):
    if isinstance(burst_list, str):
        burst_list= pd.read_csv(burst_list, parse_dates=['stime', 'etime'])
    def process_lists(list_):
        return np.array(list_.split(', '))
    burst_times= np.concatenate(burst_list.burst_timestamp.apply(process_lists).values).astype('datetime64[ns]')
    fmin= np.concatenate(burst_list.min_f_bound.apply(process_lists).values).astype(float)
    fmax= np.concatenate(burst_list.max_f_bound.apply(process_lists).values).astype(float)

    df= create_burst_numbers(pd.DataFrame({'Date_UTC':burst_times, 'fmin':fmin, 'fmax':fmax}).dropna(), burst_list,
                             time='Date_UTC', inplace=False)
    return df

def find_fmin_extension(flims, event_list='combined',
                        epoch_range=(np.timedelta64(-30, 'm'), np.timedelta64(-27, 'm')),
                        time='Date_UTC', fmin=0, sample_rate=np.timedelta64(183, 's')):
    """
    Finds the upper frequency limit of low frequency extensions during substorms by finding the average fmin within the epoch range for each substorm event.

    Parameters:
    ----------
    flims : pd.DataFrame
        Input DataFrame containing time series data of frequency limits, including the time column specified by `time`.
    event_list : pd.DataFrame
        List of substorm onset events with time column specified by `time`.
    epoch_range : tuple of np.timedelta64, optional
        Time window (start, end) time delta range on which to perform the averaging of fmin. Also used to so start grouping of substorm events. The first instance to fall within this range for each substorm starts the substorm grouping.
    time : str, optional
        Name of the column in both `flims` and `event_list` that contains the datetime information (default is 'Date_UTC').
    fmin : float or int, optional
        Minimum value to assign when the epoch is outside the defined range (default is 0). This defines the lower frequency limit of the LFE.
    sample_rate : np.timedelta64, optional
        Approximate spacing between expected samples (used to identify epochs) (default is 183 seconds).

    Returns:
    -------
    pd.DataFrame
        Modified `flims` DataFrame with adjusted `fmax` and `fmin` to capture the LFE
    """
    try:
        from progressbar import progressbar
    except ImportError:
        def progressbar(*args, **kwargs):
            return args[0]
    from .reading_tools import find_closest
    import warnings
    flims = flims.copy()
    flims.sort_values(time, inplace=True)

    # # Find the closest event time from event_list for each flim entry
    # flims['closest'] = find_closest(flims[time].values, event_list[time].values, return_index=False)

    # # Calculate time difference (epoch) from the closest event
    # flims['Epoch'] = (flims[time] - flims['closest']).values.astype('timedelta64[ns]')
    flims[event_list]= flims[event_list]*(1e9*60)*np.timedelta64(1, 'ns')

    # Identify rows within the epoch window
    ind = (flims[event_list] >= epoch_range[0]) & (flims[event_list] <= epoch_range[0] + sample_rate * 1.5)

    # Assign group identifiers to contiguous windows that meet the criteria
    flims['grp'] = np.cumsum(ind & ~ind.shift(fill_value=False))

    # Normalize group numbering to start from 0
    flims['grp'] = flims.grp.map({val: i for i, val in enumerate(flims.grp.unique())})

    # Identify where group changes
    flims['ind'] = flims['grp'].diff().astype(bool)

    flims[event_list+'_fmin']= flims['fmin']
    flims[event_list+'_fmax']= flims['fmax']

    # Set fmax = fmin for entries outside the epoch range
    flims.loc[~((flims[event_list] >= epoch_range[0]) & (flims[event_list] <= epoch_range[1])), event_list+'_fmax'] = fmin

    # Compute the median fmin within each event group and assign to fmax
    for grp in progressbar(flims.grp.unique(), max_value=flims.grp.max(),
                           prefix='Looping through event groups: '):
        if grp == 0:
            continue
        ind = (flims.grp == grp) & (flims[time] <= flims.loc[flims.grp == grp, time].values[0] + np.diff(epoch_range)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # ignoring warnings produced when there are only nans
            flims.loc[flims.grp == grp, [event_list+'_fmax']] = np.nanmedian(flims.loc[ind, event_list+'_fmin'])

    # Final adjustment: take the min of fmin and fmax to finalize the output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # ignoring warning produced when there are only nans
        flims[event_list+'_fmin'] = np.nanmin(flims[[event_list+'_fmin', event_list+'_fmax']], axis=1)

    return flims.drop(columns=['ind', 'grp'])

def find_fmax_extension(flims, event_list,
                        epoch_range=(np.timedelta64(-30, 'm'), np.timedelta64(-27, 'm')),
                        time='Date_UTC', fmax=1100, sample_rate=np.timedelta64(183, 's')):
    """
    Finds the lower frequency limit of high frequency extensions during substorms by finding the average fmin within the epoch range for each substorm event.

    Parameters:
    ----------
    flims : pd.DataFrame
        Input DataFrame containing time series data of frequency limits, including the time column specified by `time`.
    event_list : pd.DataFrame
        List of substorm onset events with time column specified by `time`.
    epoch_range : tuple of np.timedelta64, optional
        Time window (start, end) time delta range on which to perform the averaging of fmin. Also used to so start grouping of substorm events. The first instance to fall within this range for each substorm starts the substorm grouping.
    time : str, optional
        Name of the column in both `flims` and `event_list` that contains the datetime information (default is 'Date_UTC').
    fmin : float or int, optional
        Minimum value to assign when the epoch is outside the defined range (default is 0). This defines the lower frequency limit of the LFE.
    sample_rate : np.timedelta64, optional
        Approximate spacing between expected samples (used to identify epochs) (default is 183 seconds).

    Returns:
    -------
    pd.DataFrame
        Modified `flims` DataFrame with adjusted `fmax` and `fmin` to capture the HFE
    """
    try:
        from progressbar import progressbar
    except ImportError:
        def progressbar(*args, **kwargs):
            return args[0]
    from .reading_tools import find_closest
    import warnings
    flims = flims.copy()
    flims.sort_values(time, inplace=True)

    # # Find the closest event time from event_list for each flim entry
    # flims['closest'] = find_closest(flims[time].values, event_list[time].values, return_index=False)

    # # Calculate time difference (epoch) from the closest event
    # flims['Epoch'] = (flims[time] - flims['closest']).values.astype('timedelta64[ns]')

    flims[event_list]= flims[event_list]*(1e9*60)*np.timedelta64(1, 'ns')

    # Identify rows within the epoch window
    ind = (flims[event_list] >= epoch_range[0]) & (flims[event_list] <= epoch_range[0] + sample_rate * 1.5)

    # Assign group identifiers to contiguous windows that meet the criteria
    flims['grp'] = np.cumsum(ind & ~ind.shift(fill_value=False))

    # Normalize group numbering to start from 0
    flims['grp'] = flims.grp.map({val: i for i, val in enumerate(flims.grp.unique())})

    # Identify where group changes
    flims['ind'] = flims['grp'].diff().astype(bool)

    flims[event_list+'_fmin']= flims['fmin']
    flims[event_list+'_fmax']= flims['fmax']

    # Set fmax = fmin for entries outside the epoch range
    flims.loc[~((flims[event_list] >= epoch_range[0]) & (flims[event_list] <= epoch_range[1])), event_list+'_fmin'] = fmax

    # Compute the median fmin within each event group and assign to fmax
    for grp in progressbar(flims.grp.unique(), max_value=flims.grp.max(),
                           prefix='Looping through event groups: '):
        if grp == 0:
            continue
        ind = (flims.grp == grp) & (flims[time] <= flims.loc[flims.grp == grp, time].values[0] + np.diff(epoch_range)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # ignoring warnings produced when there are only nans
            flims.loc[flims.grp == grp, [event_list+'_fmin']] = np.nanmedian(flims.loc[ind, event_list+'_fmax'])

    # Final adjustment: take the min of fmin and fmax to finalize the output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # ignoring warning produced when there are only nans
        flims[event_list+'_fmax'] = np.nanmax(flims[[event_list+'_fmin', event_list+'_fmax']], axis=1)

    return flims.drop(columns=['ind', 'grp'])

if __name__=='__main__':
    file= '../../Example_Data/wi_wa_rad1_l3_akr_20030101_v01.csv'
    df= pd.read_csv(file, parse_dates=['datetime_ut'])
    # -1 represents nan values
    df.replace(-1, np.nan, inplace=True)
    # Creating unique sweeps for each unique datetime
    create_sweeps(df, time='datetime_ut', inplace=True)
    # Linear Interpolation between frequencies
    df_lin= linear_segments(df, time='datetime_ut', frequency='freq', flux='akr_flux_si_1au', preserve_cols=['datetime_ut'])
    # Integrate across fix limits
    df_int1= integrate(df_lin, flimits=(20, 1000))
    # Creating variable flims
    sweeps= np.unique(df.SWEEP)
    freqs= np.unique(df.freq)
    flims= np.random.choice(freqs, (len(sweeps), 2))
    f_min, f_max= np.min(flims, axis=1), np.max(flims, axis=1)
    flims= pd.DataFrame({'fmin':f_min, 'fmax':f_max, 'SWEEP':sweeps})
    # Integrate aross variable limits
    df_int2= integrate(df_lin, flimits=flims, sweep='SWEEP', fmin='fmin', fmax='fmax')
    df_int2= df_int2.merge(flims, on='SWEEP')
