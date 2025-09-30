from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

import sys
import numpy as np
import pandas as pd
import os
import time

class ReadError(Exception):
    pass
def validinput(inputstr, positive_answer, negative_answer):
    """
    Prompt the user for a valid input and return a boolean based on the response.

    Parameters
    ----------
    inputstr : str
        The prompt string displayed to the user.
    positive_answer : str
        The positive answer expected from the user.
    negative_answer : str
        The negative answer expected from the user.

    Returns
    -------
    bool
        True if the user's input matches the positive_answer,
        False if the user's input matches the negative_answer.

    Example
    -------
    >>> validinput('Continue? (y/n)', 'y', 'n')
    """
    answer = input(inputstr + '\n').lower()  # Get user input and convert to lowercase
    if answer == positive_answer:
        return True
    elif answer == negative_answer:
        return False
    else:
        print('Invalid response should be either ' + str(positive_answer) + ' or ' + str(negative_answer))
        # Recursively prompt again until valid input is given
        return validinput(inputstr, positive_answer, negative_answer)

def download_progress_hook(count, block_size, total_size):
    """
    Report hook to display a progress bar for downloading.

    Parameters
    ----------
    count : int
        Current block number being downloaded.
    block_size : int
        Size of each block (in bytes).
    total_size : int
        Total size of the file (in bytes).
    """
    downloaded_size = count * block_size  # Calculate downloaded size
    percentage = min(100, downloaded_size * 100 / total_size)  # Calculate percentage

    # Create a simple progress bar
    progress_bar = f"\rDownloading: {percentage:.2f}% [{downloaded_size}/{total_size} bytes]"

    # Update the progress on the same line
    sys.stdout.write(progress_bar)
    sys.stdout.flush()

    # When download is complete
    if downloaded_size >= total_size:
        print("\nDownload complete!")

def download(url, file_name, progress=True):
    """
    Download a file from a URL with optional progress reporting.

    Parameters
    ----------
    url : str
        The URL to download from.
    file_name : str
        The local file path to save the download.
    progress : bool, optional
        Whether to show a progress bar (default True).

    Returns
    -------
    tuple
        The return value from urlretrieve, or None if download fails.
    """
    if progress:
        print(file_name)
        reporthook = download_progress_hook
    else:
        reporthook = lambda count, block_size, total_size: None
    try:
        return urlretrieve(url, file_name, reporthook=reporthook)
    except HTTPError:
        # Log failed URLs to a file
        with open('./failed_urls.txt', "a", encoding="utf-8") as f:
            f.write(url + "\n")   # add newline after text
    except (TimeoutError, URLError): #  retries if there's connection issues
        time.sleep(60) # Sleep to wait 1 minute between retries
        return download(url, file_name, progress)

def cdf2pandas(file):
    """
    Convert a CDF file to a pandas DataFrame.

    Parameters
    ----------
    file : str
        Path to the CDF file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date_UTC, freq, snr_db, akr_flux_si_1au
    """
    from cdflib import CDF, cdfepoch
    data = CDF(file)  # Open the CDF file
    epoch = data['Epoch']
    freq = data['Frequency']
    flux = data['FLUX_DENSITY']
    snr = data['SNR']
    n_epoch, n_freq = flux.shape  # Get shape

    # Broadcast epochs and frequencies to match flux shape
    epoch = np.repeat(epoch, n_freq)         # each epoch repeated n_freq times
    freq = np.tile(freq, n_epoch)            # frequency cycle repeats for each epoch

    datetimes = cdfepoch.to_datetime(epoch).astype('datetime64[ns]')  # Convert epoch to datetime
    df = pd.DataFrame({'Date_UTC': datetimes, 'freq': freq,
                      'snr_db': snr.flatten(),
                      'akr_flux_si_1au': flux.flatten()})
    return df

def waters_file(date):
    """
    Construct the download URL and filename for a given date.

    Parameters
    ----------
    date : pd.Timestamp
        The date for which to construct the URL and filename.

    Returns
    -------
    str
        The full URL to the file.
    """
    if date < pd.Timestamp('1999-01-01'):
        url = f'https://maser.obspm.fr/doi/10.25935/wxv0-vr90/content/csv/{date.year}/{date.month:02d}/'
        filename = f'wi_wa_rad1_l3_akr_{date.year}{date.month:02d}{date.day:02d}_v01.csv'
    else:
        url = f'https://maser.obspm.fr/doi/10.25935/wxv0-vr90/content/cdf/{date.year}/{date.month:02d}/'
        filename = f'wi_wa_rad1_l3_akr_{date.year}{date.month:02d}{date.day:02d}_v01.cdf'
    return url + filename

def load_file(file):
    """
    Load a Waters data file (CDF or CSV) into a pandas DataFrame.

    Parameters
    ----------
    file : str
        Path to the file.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns.
    """
    if file.endswith('.cdf'):
        try:
            df = cdf2pandas(file)
        except ValueError:
            # Log failed URLs to a file
            with open('./failed_files.txt', "a", encoding="utf-8") as f:
                f.write(file + "\n")   # add newline after text
            raise ReadError(f'Problem file: {file}')
        df.loc[df.akr_flux_si_1au<0, ['akr_flux_si_1au']]=np.nan
        df.loc[df.snr_db<0, ['snr_db']]=np.nan
        for col in ['freq', 'snr_db', 'akr_flux_si_1au']:
            df[col]= df[col].astype('float64')
    elif file.endswith('.csv'):
        df = pd.read_csv(file, parse_dates=['datetime_ut'])
        df.rename(columns={'datetime_ut': 'Date_UTC'}, inplace=True)
        df.loc[df.akr_flux_si_1au<0, ['akr_flux_si_1au']]=np.nan
        df.loc[df.snr_db<0, ['snr_db']]=np.nan
    else:
        raise ValueError('File format not recognized, must be .cdf or .csv')
    return df

def combine_waters(tmp_path='./Waters_tempfiles', hdf_path='./Waters.hdf5', remove_temp=True):
    """
    Combine Waters data files (CDF and CSV) from a temporary directory into a single HDF5 file.

    Parameters
    ----------
    tmp_path : str, optional
        Path to the temporary directory containing Waters files (default './Waters_tempfiles').
    hdf_path : str, optional
        Path to the output HDF5 file (default './Waters.hdf5').
    remove_temp : bool, optional
        Whether to remove the temporary directory after combining (default True).

    Returns
    -------
    str
        Path to the combined HDF5 file.
    """
    import glob
    import shutil
    # Try to import progressbar for progress indication; if unavailable, use a dummy function
    try:
        from progressbar import progressbar
    except ImportError:
        def progressbar(*args, **kwargs):
            return args[0]
    # Find all .cdf and .csv files in the temporary directory
    files = glob.glob(f'{tmp_path}/*.cdf') + glob.glob(f'{tmp_path}/*.csv')
    files.sort()  # Sort files for consistent order
    for file in progressbar(files, prefix='Combining files: '):
        try:
            # Load each file and append to the HDF5 file under the 'main' key
            load_file(file).to_hdf(hdf_path, key='main', mode='a', append=True, format='t', data_columns=True)
        except ReadError:
            # Skip files that cannot be read
            pass
    if remove_temp:
        # shutil.rmtree(tmp_path)  # Clean up temp files if requested
        pass
    return hdf_path  # Return the path to the combined HDF5 file

def download_waters(start_date, end_date, hdf_path='./Waters.hdf5', progress=True, parallel=True):
    """
    Download Waters AKR data files for a date range, convert to DataFrame, and save to HDF5.

    Parameters
    ----------
    start_date : str or pd.Timestamp
        Start date for download.
    end_date : str or pd.Timestamp
        End date for download.
    parallel : bool, optional
        Whether to download files in parallel (default True).
    hdf_path : str, optional
        Path to save the combined HDF5 file.
    progress : bool, optional
        Whether to show download progress.

    Returns
    -------
    str or None
        Path to the HDF5 file if hdf_path is set, else None.
    """
    # Check if HDF5 file exists and prompt user if it does
    if hdf_path:
        if os.path.isfile(hdf_path):
            if not validinput('file already exists and more Waters data will be added which can lead to duplication of data continue? (y/n)', 'y', 'n'):
                raise ValueError('User Cancelled Download, Alter file name or path or remove or move the existing file and retry')
    os.makedirs('./Waters_tempfiles/', exist_ok=True)  # Create temp directory for downloads
    dates = pd.date_range(start_date, end_date, freq='D')  # Generate date range

    if parallel:
        from multiprocessing import Pool
        with Pool(4) as p:
            urls = p.map(waters_file, dates)  # Get URLs for all dates
            download_args = [(url, './Waters_tempfiles/' + url.split('/')[-1], progress) for url in urls]
        from joblib import Parallel, delayed
        Parallel(n_jobs=12, backend='threading')(delayed(download)(*args) for args in download_args)
    else:
        urls = []
        for date in dates:
            urls.append(waters_file(date))
        for url in urls:
            download(url, './Waters_tempfiles/' + url.split('/')[-1], progress)

    # If HDF5 path is set, load files and append to HDF5
    if hdf_path:
        hdf_path= combine_waters('./Waters_tempfiles', hdf_path=hdf_path, remove_temp=True)


    # Print doi information
    print('This data was obtained from https://doi.org/10.25935/wxv0-vr90 where full dataset citation can be found')

    # Report failed downloads
    with open('./failed_urls.txt', "r", encoding="utf-8") as f:
        failed_urls = f.read().splitlines()
        print('The following files could not be found online:\n' + '\n'.join([url.split('/')[-1] for url in failed_urls]))
    os.remove('./failed_urls.txt')

    # Report failed file reads
    with open('./failed_files.txt', "r", encoding="utf-8") as f:
        failed_urls = f.read().splitlines()
        print('The following files could not be read (likely incompatable variable dimensions):\n' + '\n'.join([url.split('/')[-1] for url in failed_urls]))
    os.remove('./failed_files.txt')
    if hdf_path:
        return hdf_path

def download_fogg(path='./', progress=True):
    print('This data was obtained from https://doi.org/10.25935/ayzp-1833 where full dataset citation can be found')
    download('https://maser.obspm.fr/doi/10.25935/ayzp-1833/content/fogg_akr_burst_list_1995_2004.csv',
             path+'fogg_akr_burst_list_1995_2004.csv', progress=progress)

    data= pd.read_csv(path+'fogg_akr_burst_list_1995_2004.csv', parse_dates=['stime', 'etime'])
    data.sort_values('stime', inplace=True)
    data['Burst_Number']= np.arange(len(data)).astype(int)+1
    data.to_csv(path+'fogg_akr_burst_list_1995_2004.csv', index=False)
    return path+'fogg_akr_burst_list_1995_2004.csv'