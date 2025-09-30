import struct  # Import struct module for handling binary data
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from functools import reduce  # Import reduce function for merging DataFrames
import os  # Import os for file path operations

def read_l2_hres(filepath, verbose=True):
    """
    Reads a Wind/Waves L2 high-resolution data file and returns metadata and data as DataFrames.

    Parameters
    ----------
    filepath : str
        Path to the binary data file.
    verbose : bool, optional
        If True, prints progress information while reading the file. Default is True.

    Returns
    -------
    df_headers : pd.DataFrame
        DataFrame containing header metadata for each sweep.
    df_data : pd.DataFrame
        DataFrame containing expanded frequency, intensity, and time values.
    nsweep : int
        Number of sweeps processed.

    Example
    -------
    >>> headers, data, nsweeps = read_l2_hres('data.bin')
    """
    # Define the fields and data types for the header
    header_fields = ("P_FIELD", "JULIAN_DAY_B1", "JULIAN_DAY_B2", "JULIAN_DAY_B3", "MSEC_OF_DAY",
                     "RECEIVER_CODE", "JULIAN_SEC_FRAC", "YEAR", "MONTH", "DAY",
                     "HOUR", "MINUTE", "SECOND", "JULIAN_SEC_FRAC_2",
                     "ISWEEP", "IUNIT", "NPBS", "SUN_ANGLE", "SPIN_RATE", "KSPIN", "MODE", "LISTFR", "NFREQ",
                     "ICAL", "IANTEN", "IPOLA", "IDIPXY", "SDURCY", "SDURPA",
                     "NPALCY", "NFRPAL", "NPALIF", "NSPALF", "NZPALF")
    header_dtype = '>bbbbihLhhhhhhfihhffhhhhhhhhffhhhhh'
    
    header_list = []  # List to store header data
    data_list = []  # List to store data values
    nsweep = 1  # Initialize sweep counter
    freqs = []  # List to store frequency values
    
    # Open the binary file for reading
    with open(filepath, 'rb') as frb:
        while True:
            try:
                if verbose:
                    print(f"Reading sweep #{nsweep}")  # Print progress if verbose
                position = frb.tell()  # Store current file position
                block = frb.read(4)  # Read the first 4 bytes to get the octet count
                if len(block) == 0:
                    break
                loctets1 = struct.unpack('>i', block)[0]  # Unpack integer value
                block = frb.read(80)  # Read 80-byte header
                if len(block) != 80:
                    if verbose:
                        print("Incomplete header read.")  # Handle incomplete header
                    break
                
                # Unpack the header using the defined format
                header_values = struct.unpack(header_dtype, block)
                header_dict = dict(zip(header_fields, header_values))  # Map values to field names
                header_dict.update({'SWEEP': nsweep})  # Assign sweep number
                header_list.append(header_dict)  # Append to header list
                
                # Extract key header parameters
                npalf, nspal, nzpal = header_dict["NPALIF"], header_dict["NSPALF"], header_dict["NZPALF"]
                
                # Read frequency values
                block = frb.read(4 * npalf)
                freq = struct.unpack('>' + 'f' * npalf, block)
                
                # Read intensity values for S/SP
                block = frb.read(4 * npalf * nspal)
                Vspal = struct.unpack('>' + 'f' * npalf * nspal, block)
                
                # Read time values for S/SP
                block = frb.read(4 * npalf * nspal)
                Tspal = struct.unpack('>' + 'f' * npalf * nspal, block)
                
                # Read intensity values for Z
                block = frb.read(4 * npalf * nzpal)
                Vzpal = struct.unpack('>' + 'f' * npalf * nzpal, block)
                
                # Read time values for Z
                block = frb.read(4 * npalf * nzpal)
                Tzpal = struct.unpack('>' + 'f' * npalf * nzpal, block)
                
                # Read the trailing 4-byte integer and verify consistency
                block = frb.read(4)
                loctets2 = struct.unpack('>i', block)[0]
                
                if loctets2 != loctets1:
                    print("Mismatch in sweep octets.")  # Detect file corruption
                    break
                
                # Expand frequency values to match corresponding time values
                freq = np.repeat(freq, len(Tzpal)/len(freq))
                freqs.append(freq)
                
                # Store extracted values in data_list
                for i in range(len(freq)):
                    data_list.append({
                        "FREQ": round(freq[i]),  # Rounded frequency value
                        "VSPAL": Vspal[i] if i < len(Vspal) else None,  # Intensity value for S/SP
                        "VS2PAL": Vspal[i] if i < len(Vspal) else None,  # Intensity value for second S/SP
                        "VZPAL": Vzpal[i] if i < len(Vzpal) else None,  # Intensity value for Z
                        "TSPAL": Tspal[i] if i < len(Tspal) else None,  # Time value for S/SP
                        "TS2PAL": Tspal[i] if i < len(Tspal) else None,  # Time value for second S/SP
                        "TZPAL": Tzpal[i] if i < len(Tzpal) else None,  # Time value for Z
                        'SWEEP': nsweep  # Sweep number
                    })
                
                nsweep += 1  # Increment sweep counter
            except struct.error as e:
                print(f"Binary unpacking error: {e}")  # Handle struct unpacking errors
                break
    
    # Convert extracted data into DataFrames
    df_headers = pd.DataFrame(header_list)
    df_data = pd.DataFrame(data_list)
    
    return df_headers, df_data, nsweep  # Return metadata, data, and sweep count


def to_datetime(data):
    """
    Converts year, month, and day columns into an ISO 8601 datetime string format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'YEAR', 'MONTH', and 'DAY' columns.

    Returns
    -------
    pd.Series
        Series of date strings formatted as 'YYYY-MM-DDT00:00'.

    Example
    -------
    >>> df = pd.DataFrame({'YEAR': [2024], 'MONTH': [3], 'DAY': [1]})
    >>> to_datetime(df)
    0    2024-03-01T00:00
    dtype: object
    """
    return data['YEAR'].astype(str) + '-' + data['MONTH'].apply(lambda x: f'{int(x):02d}') + '-' + data['DAY'].apply(lambda x: f'{int(x):02d}') + 'T00:00'

def quality_flags(data):
    """
    Computes quality flags based on instrument parameters to assess data reliability.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing measurement and instrument state parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing quality checks per sweep, indicating valid or invalid conditions.

    Example
    -------
    >>> df = pd.DataFrame({'SWEEP': [1, 1, 2, 2], 'IUNIT': [3, 3, 3, 3], 'KSPIN': [0, 0, 0, 0], 'MODE': [3, 3, 3, 3],
    ...                   'IANTEN': [2, 2, 2, 2], 'IPOLA': [1, 1, 1, 1], 'IDIPXY': [1, 1, 1, 1],
    ...                   'NSPALF': [4, 4, 6, 6], 'NZPALF': [2, 2, 3, 3]})
    >>> quality_flags(df)
       SWEEP  unit_check  kspin_check  mode_check  ant_check  pol_check  eq_dipole_check  nsample_check
    0      1       True         True        True       True       True             True           True
    1      2       True         True        True       True       True             True           True
    """
    def status_check(x, val):
        return (x == val).all()  # Check if all values in the group match the expected value
    
    def check_match(group):
        return (group['NSPALF'] == 2 * group['NZPALF']).all()  # Validate sample count consistency
    
    grp = data.groupby('SWEEP')  # Group by sweep identifier
    
    # Apply quality checks across different parameters
    dfs = [
        grp['IUNIT'].apply(status_check, val=3).reset_index().rename(columns={'IUNIT': 'unit_check'}),
        grp['KSPIN'].apply(status_check, val=0).reset_index().rename(columns={'KSPIN': 'kspin_check'}),
        grp['MODE'].apply(status_check, val=3).reset_index().rename(columns={'MODE': 'mode_check'}),
        grp['IANTEN'].apply(status_check, val=2).reset_index().rename(columns={'IANTEN': 'ant_check'}),
        grp['IPOLA'].apply(status_check, val=1).reset_index().rename(columns={'IPOLA': 'pol_check'}),
        grp['IDIPXY'].apply(status_check, val=1).reset_index().rename(columns={'IDIPXY': 'eq_dipole_check'}),
        grp.apply(check_match).reset_index().rename(columns={0: 'nsample_check'})
    ]
    
    return reduce(lambda left, right: pd.merge(left, right, on='SWEEP'), dfs)  # Merge quality checks

def process(headers, data):
    """
    Processes and merges header and data information, adding timestamps and quality flags.

    Parameters
    ----------
    headers : pd.DataFrame
        DataFrame containing metadata for each sweep.
    data : pd.DataFrame
        DataFrame containing frequency and intensity values per sweep.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with merged metadata, timestamps, and quality flags.

    Example
    -------
    >>> headers = pd.DataFrame({'SWEEP': [1, 2], 'YEAR': [2024, 2024], 'MONTH': [3, 3], 'DAY': [1, 1],
    ...                         'MSEC_OF_DAY': [1000, 2000]})
    >>> data = pd.DataFrame({'SWEEP': [1, 1, 2, 2], 'TSPAL': [0.1, 0.2, 0.1, 0.2], 'TS2PAL': [0.15, 0.25, 0.15, 0.25],
    ...                      'TZPAL': [0.05, 0.15, 0.05, 0.15]})
    >>> process(headers, data)
       SWEEP  YEAR  MONTH  DAY  MSEC_OF_DAY  Date_UTC        DATETIME_S        DATETIME_SP       DATETIME_Z
    0      1  2024      3    1        1000 2024-03-01 2024-03-01 00:00:01 2024-03-01 00:00:01.150 2024-03-01 00:00:01.050
    1      2  2024      3    1        2000 2024-03-01 2024-03-01 00:00:02 2024-03-01 00:00:02.150 2024-03-01 00:00:02.050
    """
    data = data.copy()  # Create a copy of the data to avoid modifying the original DataFrame
    data = headers.merge(data, on='SWEEP')  # Merge header and data on sweep identifier
    
    # Convert date information to datetime format and add milliseconds of the day
    data['Date_UTC'] = to_datetime(data).astype('datetime64[ns]') + data['MSEC_OF_DAY'].astype('timedelta64[ms]')
    
    # Compute absolute timestamps for different time parameters
    data['DATETIME_S'] = data['Date_UTC'] + pd.to_timedelta(data['TSPAL'], 's')  # Time for S/SP
    data['DATETIME_SP'] = data['Date_UTC'] + pd.to_timedelta(data['TS2PAL'], 's')  # Time for second S/SP
    data['DATETIME_Z'] = data['Date_UTC'] + pd.to_timedelta(data['TZPAL'], 's')  # Time for Z
    
    flags = quality_flags(data)  # Compute quality flags for data validation
    flags['sweep_tag'] = flags.filter(like='_check').all(axis=1)  # Check overall sweep validity
    
    data = data.merge(flags, on='SWEEP')  # Merge quality flags with the data
    
    return data  # Return processed DataFrame

RE = 6378100  # Earth's radius in meters
au = 149597870700  # Astronomical unit in meters

def normalise(data, position=__file__.split('src')[0]+'Example_Data/WIND_Position_1994_2010_vx.hdf5',
              datetime='Date_UTC', au_normalise=['VZPAL'], inplace=False):
    """
    Normalise data columns to 1 AU using spacecraft position.

    Parameters
    ----------
    data : pd.DataFrame
        Data to normalise.
    position : str or DataFrame, optional
        Path to position of WIND data or DataFrame.
    datetime : str, optional
        Column name for datetime.
    au_normalise : list of str, optional
        List of columns to normalise to 1 AU.
    inplace : bool, optional
        If True, modifies data in place.

    Returns
    -------
    pd.DataFrame or None
        Normalised data if inplace is False, otherwise None.
    """
    import vaex as vx  # Import vaex for fast DataFrame operations
    if not inplace:
        data = data.copy()  # Work on a copy if not inplace
    data['Date_UTC'] = data[datetime]  # Ensure Date_UTC column exists
    min_time, max_time = data.Date_UTC.min().to_numpy(), data.Date_UTC.max().to_numpy()  # Get time range
    if isinstance(position, str):
        position = vx.open(position)  # Open position file if path is given
    # Select position data within time window
    position.select((position.Date_UTC >= min_time - np.timedelta64(30, 'm')) &
                    (position.Date_UTC <= max_time + np.timedelta64(30, 'm')))
    position = position.to_pandas_df(selection=True)  # Convert to pandas DataFrame
    # Interpolate radius for each data point
    data['radius'] = np.interp(data.Date_UTC.astype(np.int64),
                               position.Date_UTC.astype(np.int64),
                               position['radius'])
    if len(au_normalise):
        dist_in_au = (data['radius'] * RE) / au  # Convert radius to AU
        for col in au_normalise:
            data[f'{col}_1au'] = data[col] * dist_in_au ** 2  # Normalise column to 1 AU
    if not inplace:
        return data  # Return normalised data if not inplace

def raw2csv(date, csv_folder, dat_folder):
    """
    Convert raw .dat file to .csv, normalising and saving if not already present.
    This code perform normlisation and conversion to csv if a csv file does not exist.
    This means that it avoids unnecessary repition.

    Parameters
    ----------
    date : str or pd.Timestamp
        Date of the file to process.
    csv_folder : str
        Folder to save CSV files.
    dat_folder : str
        Folder containing .dat files.

    Returns
    -------
    str
        Path to the CSV file.
    """
    date = pd.Timestamp(date)  # Ensure date is a Timestamp
    year, month, day = date.year, date.month, date.day  # Extract year, month, day
    csv_path = f'{csv_folder}{year}/wi_wa_rad1_l2_{year}{month:02d}{day:02d}_v01.csv'
    dat_path = f'{dat_folder}{year}/wi_wa_rad1_l2_{year}{month:02d}{day:02d}_v01.dat'
    if os.path.isfile(csv_path):
        return csv_path  # Return if CSV already exists
    else:
        if not os.path.isdir(f'{csv_folder}{year}'):
            os.makedirs(f'{csv_folder}{year}')  # Create year folder if needed
        # Read, process, normalise, and save to CSV
        normalise(process(*read_l2_hres(dat_path)[:-1])).to_csv(csv_path, index=False)
    return csv_path  # Return path to saved CSV

def find_closest(A, B, return_index=False, type='absolute'):
    """
    Find closest values in B for each value in A.

    Parameters
    ----------
    A : array-like
        Array of values to match.
    B : array-like
        Array of candidate values.
    return_index : bool, optional
        If True, also return indices of closest matches.
    type : str, optional
        'absolute', 'below', or 'above' for matching logic.

    Returns
    -------
    np.ndarray or tuple
        Closest values (and indices if requested).
    """
    A = np.asarray(A)  # Convert A to numpy array
    B = np.asarray(B)  # Convert B to numpy array

    B_sorted = np.sort(B)  # Sort B for search
    B_indices = np.argsort(B)  # Indices to map sorted B to original

    idx = np.searchsorted(B_sorted, A, side="left")  # Find insertion indices

    idx_safe = np.clip(idx, 0, len(B_sorted) - 1)  # Ensure indices are in bounds

    exact_match = (idx < len(B_sorted)) & (B_sorted[idx_safe] == A)  # Find exact matches

    idx_below = np.where(exact_match, idx_safe, np.clip(idx - 1, 0, len(B_sorted) - 1))  # Index for below value

    below = np.where(A < B_sorted[0], np.datetime64('NaT'), B_sorted[idx_below])  # Below value or NaT
    above = B_sorted[idx_safe]  # Above value

    if type == 'absolute':
        closest_vals = np.where(np.abs(A - below) <= np.abs(A - above), below, above)  # Closest value
        if return_index:
            closest_inds = np.where(np.abs(A - below) <= np.abs(A - above), idx_below, idx_safe)
            closest_inds = B_indices[closest_inds]  # Map to original indices
            return closest_vals, closest_inds
        return closest_vals

    elif type == 'below':
        if return_index:
            closest_inds = np.where(A >= below, idx_below, -1)  # -1 if no valid below index
            closest_inds = np.where(below != np.datetime64('NaT'), B_indices[closest_inds], -1)
            return below, closest_inds
        return below

    elif type == 'above':
        closest_vals = np.where(A <= above, above, np.datetime64('NaT'))  # NaT if no valid above value
        if return_index:
            closest_inds = np.where(A <= above, idx_safe, -1)  # -1 if no valid above index
            closest_inds = np.where(closest_vals != np.datetime64('NaT'), B_indices[closest_inds], -1)
            return closest_vals, closest_inds
        return closest_vals

    else:
        raise ValueError("Invalid type. Choose from 'absolute', 'below', or 'above'.")
def lioupdf2csv(path='./liou.pdf'):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    # text= '\n'.join([page.extract_text() for page in reader.pages])
    def drop_index(row, i=0):
        row= row.split(' ')
        if i<998:
            return row[0][-8:]
        else:
            return row[1]
    def remove_date(row, i):
        row= row.split(' ')
        if i<998:
            new_row= [row[1]+row[2]]+[r for r in row[3:] if r!=' ']
        else:
            new_row= [row[2]+row[3]]+[r for r in row[4:] if r!=' ']
        new_row=' '.join(new_row)
        new_row=' '.join(new_row.split('-'))
        return new_row
    def format_data(d, i):
        d='.'.join(d.split(','))
        d= d.split()
        if i==842:
            d= d[:3]+ [d[3]+'.'+d[4]] + d[5:]
        elif i==1355:
            d= [d[0], d[1]+'0'+d[2]]+d[3:]
        elif i==55:
            d[2]= '.'.join(d[2].split('/'))
        # elif i==1536:
        #     d[2]= '.'.join(d[2].split(','))
        #     d[3]= '.'.join(d[3].split(','))
        return d
    columns= [reader.pages[0].extract_text().split('\n')[0][1:]]
    columns=[c for c in columns[0].split(' ') if c!=' ' and c!='']
    columns= columns[:2]+[columns[2]+columns[3]]+columns[4:]
    text= '\n'.join([page.extract_text() for page in reader.pages])
    dates=  [drop_index(row, i) for i, row in enumerate(text.split('\n')[1:])]
    data= [remove_date(row, i) for i, row in enumerate(text.split('\n')[1:])]

    data= np.array([format_data(d, i) for i, d in enumerate(data)], dtype=object)
    df= {'Date_UTC': pd.to_datetime(dates)}
    df.update({col: d for col, d in zip(columns[1:], data.T)})
    df= pd.DataFrame(df)
    for col in columns[3:]:
        df[col]= df[col].astype(float)
    df["start_time"] = pd.to_datetime(df["Date_UTC"].astype(str) + " " + df["time1"], format="%Y-%m-%d %H%M%S", errors="coerce")
    df["end_time"] = pd.to_datetime(df["Date_UTC"].astype(str) + " " + df["time2"], format="%Y-%m-%d %H%M%S", errors="coerce")
    df['Date_UTC']= df[['start_time', 'end_time']].mean(axis=1)
    df.to_csv(path.split('.pdf')[0]+'.csv', index=False)
    return path.split('.pdf')[0]+'.csv'

def read_combine_wind_position(path='./'):
    from functools import partial
    import glob
    RE*=1e-3

    func= partial(pd.read_csv, comment='#', header=0, delim_whitespace=True)
    def to_hr(x):
        val= x.split(':')
        return int(val[0])+int(val[1])/60+int(val[2])/(60*60)
    files= glob.glob(path+'*.txt')

    data= pd.concat([func(file) for file in files if file!='position_geo.txt'])

    data['Date_UTC']= ((data['yyyy'].astype(str).values.astype('datetime64[Y]')+ (data.doy-1).astype('timedelta64[D]')).astype(str)+ 'T'+data['hh:mm:ss']).values.astype('datetime64[m]')
    data['Date_UTC']= data.Date_UTC.values.astype('datetime64[ns]')
    d_lim= data.Date_UTC.min(), data.Date_UTC.max()
    geo_data= pd.read_csv('position_geo.txt', header=0, delim_whitespace=True)
    geo_data['Date_UTC']= ((geo_data['yyyy'].astype(str).values.astype('datetime64[Y]')+ (geo_data.doy-1).astype('timedelta64[D]')).astype(str)+ 'T'+geo_data['hh:mm:ss']).values.astype('datetime64[m]')
    geo_data['Date_UTC']= geo_data.Date_UTC.values.astype('datetime64[ns]')
    geo_data= geo_data.loc[(geo_data.Date_UTC>=d_lim[0])&(geo_data.Date_UTC<=d_lim[-1])]
    data= data.merge(geo_data.drop(columns=['yyyy', 'doy', 'hh:mm:ss']), on='Date_UTC')
    data['gseLT_Hr']= data.gseLT.apply(to_hr)
    data['geoLT_Hr']= data.geoLT.apply(to_hr)
    return data



if __name__=='__main__':
    filepath='../../Example_Data/wi_wa_rad1_l2_19990119_v01.dat'
    header, data, sweep= read_l2_hres(filepath)
    data_flagged= process(header, data)


    filepath='../../Example_Data/wi_wa_rad1_l2_19990118_v01.dat'
    header2, data2, sweep= read_l2_hres(filepath)
    data2_flagged= process(header2, data2)

