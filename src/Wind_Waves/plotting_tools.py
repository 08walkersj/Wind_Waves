import numpy as np
import matplotlib.pyplot as plt
# freqs= np.array([  20,   24,   28,   32,   36,   40,   44,   48,   52,   60,   72,
#      80,   92,  104,  124,  136,  148,  176,  196,  224,  256,  272,
#     332,  388,  428,  484,  540,  624,  740,  804,  940, 1040]) # These are the typical 32 frequencies measured
def AKR_mask(mean, std, threshold):
    """ Calculating the Coeffient of Variation (CV) and applying a threshold to create a mask for AKR data.
    Parameters
    ----------
    mean : np.ndarray/xarray.DataArray
        Array of mean values.
    std : np.ndarray/xarray.DataArray
        Array of standard deviation values.
    
    returns
    -------
    np.ndarray/xarray.DataArray
        Boolean mask where CV exceeds the threshold."""
    return (std/mean) > threshold

def _interp1d_no_extrap(data, max_gap=None):
    """
    Interpolate NaN values in a 1D array without extrapolating beyond the range of finite values.
    If max_gap is set, only interpolate NaN runs shorter than max_gap.

    Parameters
    ----------
    data : array-like
        1D array containing values to interpolate (NaNs will be filled).
    max_gap : int or None, optional
        Maximum length of consecutive NaNs to interpolate. Longer gaps are left as NaN.
        If None, all gaps are interpolated.

    Returns
    -------
    yy : ndarray
        Array with NaNs interpolated (where allowed).
    """
    data = np.asarray(data, dtype=float)  # Ensure input is a float array
    x = np.arange(data.size)           # Create index array for interpolation
    m = np.isfinite(data)              # Boolean mask of finite (non-NaN) values
    if m.sum() < 2:
        return data.copy()             # Not enough points to interpolate, return original

    left, right = m.argmax(), (data.size - 1) - m[::-1].argmax()  # Find first and last finite indices
    core = slice(left, right + 1)   # Slice covering the core region with finite values

    yy = data.copy()                   # Copy input array for output
    xc = x[core]                    # Indices within core region
    mc = m[core]                    # Mask of finite values within core

    if (~mc).any():                 # If there are NaNs in the core region
        interp_vals = np.interp(xc[~mc], xc[mc], yy[core][mc])  # Interpolate missing values

        if max_gap is None:
            yy[core][~mc] = interp_vals     # Fill all missing values in core
        else:
            nan_runs = np.where(~mc)[0]     # Indices of NaNs in core
            splits = np.split(nan_runs, np.where(np.diff(nan_runs) > 1)[0] + 1)  # Split into contiguous runs
            for run in splits:
                if len(run) <= max_gap:     # Only fill gaps shorter than max_gap
                    yy[core][run] = interp_vals[np.isin(nan_runs, run)]
    return yy

def gap_fill(data, max_gap_time = int((183 / 3) * 3), max_gap_freq = 70):
    """
    Fill gaps in a 2D spectrogram array (time × frequency) using 1D interpolation.
    Gaps are filled along time for each frequency, and along frequency for each time.
    Only gaps shorter than max_gap_time or max_gap_freq are filled.

    Parameters
    ----------
    data : array-like
        2D array (time × frequency) with gaps (NaNs) to fill.
    max_gap_time : int, optional
        Maximum gap size (in time axis) to interpolate per frequency.
    max_gap_freq : int, optional
        Maximum gap size (in frequency axis) to interpolate per time.

    Returns
    -------
    filled : ndarray
        Array with gaps filled where allowed.
    """
    data = np.asarray(data, dtype=float)    # Ensure input is a float array
    out = data.copy()                       # Copy input for output

    with np.errstate(invalid='ignore'):
        L = np.log10(out)                   # Take log10 for interpolation (preserves ratios)

    # Pass 1: interpolate along time axis (per frequency)
    for r in range(L.shape[0]):
        row = L[r, :]                       # Extract row (frequency slice)
        if np.isfinite(row).sum() >= 2:     # Only interpolate if at least 2 finite values
            L[r, :] = _interp1d_no_extrap(row, max_gap=max_gap_time)

    # Pass 2: interpolate along frequency axis (per time)
    for c in range(L.shape[1]):
        col = L[:, c]                       # Extract column (time slice)
        n_finite = np.isfinite(col).sum()   # Count finite values
        frac = n_finite / col.size          # Fraction of finite values (not used)
        if n_finite >= 2:                   # Only interpolate if at least 2 finite values
            L[:, c] = _interp1d_no_extrap(col, max_gap=max_gap_freq) # 70 points ≈ 140kHz

    return 10 ** L                          # Return filled array in original scale


def attach_corner_label(ax, fig, text, location='below left', offset=(0, 0.04)):
    """
    Attach a label to a subplot corner that updates its position when the figure is redrawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot axis to attach the label to.
    fig : matplotlib.figure.Figure
        The figure object containing the subplot.
    text : str
        The label text to display.
    location : str, optional
        Corner location for the label ('below left' or 'above left').
    offset : tuple of float, optional
        Offset (x, y) in figure coordinates from the chosen corner.

    Returns
    -------
    label : matplotlib.text.Text
        The created text object (can be modified later).
    """

    # Create the label text at a dummy position; will be updated on draw
    label = fig.text(0, 0, text, ha='left', va='top')

    def update_label(event):
        """
        Update the label position and font size when the figure is redrawn.

        Parameters
        ----------
        event : matplotlib.backend_bases.DrawEvent
            The draw event triggered by the figure canvas.
        """
        bbox = ax.get_position()  # Get the axis bounding box in figure coordinates

        # Determine label position and vertical alignment based on location
        if location == 'below left':
            x = bbox.x0 + offset[0]  # Left edge plus x offset
            y = bbox.y0 - offset[1]  # Bottom edge minus y offset
            va = 'top'               # Align text to top
        elif location == 'above left':
            x = bbox.x0 + offset[0]  # Left edge plus x offset
            y = bbox.y1 + offset[1]  # Top edge plus y offset
            va = 'bottom'            # Align text to bottom
        else:
            raise ValueError("Use 'below left' or 'above left' for location.")

        # Match font size to the first x-tick label (if available)
        if ax.get_xticklabels():
            xtick_size = ax.get_xticklabels()[0].get_size()
            label.set_fontsize(xtick_size)

        # Update label position and vertical alignment
        label.set_position((x, y))
        label.set_va(va)

    # Connect the update function to the figure's draw event
    fig.canvas.mpl_connect('draw_event', update_label)

    return label  # Return the label object for further modification if needed

def multi_xlabels(ax, start_strs=[], keep_original=''):
    """
    Add multiple lines of x-axis tick labels using custom functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to modify x tick labels for.
    start_strs : list of str, optional
        Strings to prepend to the first tick label.
    keep_original : str, optional
        If non-empty, keep the original tick label text as the first line.

    Returns
    -------
    None
    """
    # Check if there are label functions defined for the axis
    if len(ax.label_funcs):
        ts = []  # List to hold new tick label strings
        for i, l in enumerate(ax.get_xticklabels()):
            # For the first tick, optionally remove an existing start string label
            if len(start_strs) and not i:
                try:
                    ax.tick_start_str.remove()  # Remove previous start string label if present
                except:
                    pass
                pos = l.get_position()  # Get tick label position
                x = ''  # Initialize label string
                # Set vertical alignment based on position (not used here)
                if pos[1]:
                    va = 'bottom'
                else:
                    va = 'top'
            x = ''  # Initialize label string for each tick
            # Optionally keep the original tick label as the first line
            if len(keep_original):
                x = l.get_text().split('\n')[0] + '\n'
            # Apply each label function to the tick value and join results as new lines
            ts.append(x + '\n'.join([f'{np.round(func(l._x), 2)}' for func in ax.label_funcs]))
        # Set the new tick labels on the axis
        ax.set_xticklabels(ts)

def vx_bins2binby(start, end, step=1):
    """
    Generate binning parameters for Vaex's `binby` function based on the given start, end, and step.

    Parameters:
        start (float): The starting value of the binning range.
        end (float): The ending value of the binning range.
        step (float, optional): The step size for binning. Defaults to 1.

    Returns:
        dict: A dictionary containing binning parameters suitable for `binby` function in Vaex.
            - 'limits': Tuple containing the lower and upper limits of the binning range, centered on whole number bins.
            - 'shape': Number of bins calculated based on the provided range and step size.

    Example:
        >>> vx_bins2binby(0, 10, 2)
        {'limits': (-1.0, 11.0), 'shape': 6}
    """
    return {'limits': (start-step/2, end + step/2), 'shape': int((end-start)/step) + 1}

def spectragram(raw, ax=False, gap_filling='fill', groupby='SWEEP', frequency='freq', datetime= 'DATETIME_Z', flux='AMPL_Z',
                frequencies= np.arange(20, 1041, 4), frequency_step=4, frequency_bandwidth=3,
                start=False, end=False, akr_mask=False, akr_threshold=0.1, vmin= False, vmax= False, no_data='white', gap_fill_kwargs={'max_gap_time' : int((183 / 3) * 3), 'max_gap_freq' : 70}, **pcolormesh_kwargs):
    """
    Build a spectrogram of averaged flux vs. time/frequency, with optional gap handling and plotting.

    Parameters
    ----------
    raw : vaex DataFrame
        Input dataset containing sweep, frequency, datetime, and flux columns.
    ax : matplotlib.axes.Axes or False, optional
        If provided, plot the spectrogram on this axis; otherwise return the grid arrays.
    gap_filling : {'keep', 'ignore', 'fill'}, optional
        Strategy for handling missing time/frequency bins. 'keep' leaves NaNs, 'ignore' uses native
        frequencies only, 'fill' interpolates gaps via `gap_fill`.
    groupby : str, optional
        Column name used to group sweeps along the time axis.
    frequency : str, optional
        Column name for frequency values in `raw`.
    datetime : str, optional
        Column name for timestamp values in `raw`.
    flux : str, optional
        Column name for the measurement to plot.
    frequencies : array-like, optional
        Target frequency grid (Hz or kHz) when gap_filling is 'keep' or 'fill'.
    frequency_step : float, optional
        Step size for the target frequency grid.
    frequency_bandwidth : float, optional
        Width of each frequency bin.
    start, end : scalar or False, optional
        Optional datetime bounds to subset sweeps.
    akr_mask : bool, optional
        Apply AKR coefficient-of-variation mask before plotting.
    akr_threshold : float, optional
        Threshold for AKR_mask.
    vmin, vmax : float or False, optional
        Log-scale color limits for plotting; computed from data if False.
    no_data : str, optional
        Color used for NaN regions in the plot.
    gap_fill_kwargs : dict, optional
        Arguments passed to `gap_fill` when `gap_filling='fill'`.
    **pcolormesh_kwargs :
        Extra arguments forwarded to `ax.pcolormesh`.

    Returns
    -------
    tuple
        If `ax` is provided, returns (QuadMesh, (pixel time edges, pixel frequency edges, pixel values));
        otherwise returns (pixel time edges, pixel frequency edges, pixel value).
    """
    import matplotlib.colors as mp_colors

    # Enforce supported gap filling options early
    if not gap_filling in ['keep', 'ignore', 'fill']:
        raise ValueError("Choice of gap filling method not understood please choose one of the following: 'keep', 'ignore' or 'fill'")

    # Restrict dataset to desired sweep range (optionally by date bounds)
    if start and end:
        raw.select((raw[datetime]>=start)&(raw[datetime]<=end))
        sweep_start, sweep_end= raw[groupby].min(selection=True)-1, raw[groupby].max(selection=True)+1
        raw.select((raw[groupby]>=sweep_start+1)&(raw[groupby]<=sweep_end-1))
    else:
        sweep_start, sweep_end= raw[groupby].min(), raw[groupby].max()
        raw.select((raw[groupby]>=sweep_start+1)&(raw[groupby]<=sweep_end-1))

    # Compute sweep and time bin edges
    sweep_binby= vx_bins2binby(raw[groupby].min(selection=True), raw[groupby].max(selection=True), 1)
    t1, t2= raw[datetime].min(raw[groupby], **sweep_binby), raw[datetime].max(raw[groupby], **sweep_binby)
    time_edges= np.concatenate(np.vstack([t1, t2]).T)
    if gap_filling in ['keep', 'fill']:
        # Create regular frequency bins and initialize container
        freq_binby= vx_bins2binby(np.min(frequencies), np.max(frequencies), frequency_step)
        freq_edges = np.vstack([frequencies-frequency_bandwidth/2, frequencies+frequency_bandwidth/2])
        freq_edges = np.concatenate(freq_edges.T)
        values = np.full((len(freq_edges)-1, len(time_edges)-1), np.nan)
        # Bin mean flux into regular grid
        rw = raw.mean(raw[flux], binby=[raw[frequency], raw[groupby]], limits=[freq_binby['limits'], sweep_binby['limits']],
                    shape=(freq_binby['shape'], sweep_binby['shape']), array_type='xarray', selection=True)
        if akr_mask:
            rw= rw.where(AKR_mask(rw, raw.std(raw[flux], binby=[raw[frequency], raw[groupby]], limits=[freq_binby['limits'], sweep_binby['limits']],
                            shape=(freq_binby['shape'], sweep_binby['shape']), array_type='xarray', selection=True), akr_threshold))
        values[::2, ::2] = rw.values


    elif gap_filling == 'ignore':
        # Keep native irregular frequencies and map them into contiguous bins
        freqs = np.sort(raw[frequency].unique(selection=True))
        freq_edges = np.concatenate(([freqs[0]-np.diff(freqs)[0]/2], freqs[1:]-np.diff(freqs)/2, [freqs[-1]+np.diff(freqs)[-1]/2]))
        freq_list = np.sort(raw[frequency].unique())
        mapper = {freq: i for i, freq in enumerate(freqs)}
        for freq in freq_list:
            if freq not in mapper:
                mapper[freq] = -1
        raw['frequency_bin'] = raw[frequency].map(mapper)
        freq_bin_binby = vx_bins2binby(0, raw['frequency_bin'].max(), 1)
        values = np.full((len(freqs), len(time_edges)-1), np.nan)
        # Bin mean flux on irregular frequency bins
        rw = raw.mean(raw[flux], binby=[raw.frequency_bin, raw[groupby]], limits=[freq_bin_binby['limits'], sweep_binby['limits']],
                      shape=(freq_bin_binby['shape'], sweep_binby['shape']), array_type='xarray', selection=True)


        if akr_mask:
            rw= rw.where(AKR_mask(rw, raw.std(raw[flux], binby=[raw.frequency_bin, raw[groupby]], limits=[freq_bin_binby['limits'], sweep_binby['limits']],
                            shape=(freq_bin_binby['shape'], sweep_binby['shape']), array_type='xarray', selection=True), akr_threshold))
        values[:, ::2] = rw.values

    # Clean zeros and optionally interpolate gaps
    values[values == 0] = np.nan
    if gap_filling=='fill':
        values= gap_fill(values, **gap_fill_kwargs)
    if ax :
        # Derive color limits on log scale if not provided
        vmax_ = np.round(np.nanquantile(np.log10(values), .9))
        vmin_ = max([vmax_-1.5, np.round(np.nanmin(np.log10(values)))-1])
        if not vmax:
            vmax= 10**vmax_
        if not vmin:
            vmin= 10**vmin_
        # Render pcolormesh with masked NaNs
        pc= ax.pcolormesh(time_edges, freq_edges, np.ma.masked_invalid(values), 
                                norm=mp_colors.LogNorm(vmin=vmin, vmax=vmax), **pcolormesh_kwargs)
        pc.get_cmap().set_bad(no_data)
        return pc, (time_edges, freq_edges, values)
    else:
        return time_edges, freq_edges, values

def mlt_formatter(x, pos):
    return f"{x+24:.0f}" if x < 0 else f"{x:.0f}"

def combine_handles_labels(*axes):
    """
    Collects legend handles and labels from multiple matplotlib Axes objects
    and returns a combined list without duplicates.
    
    Parameters
    ----------
    *axes : matplotlib.axes.Axes
        Any number of subplot Axes.
    
    Returns
    -------
    handles : list
        Combined list of legend handles.
    labels : list
        Combined list of legend labels.
    """
    handles, labels = [], []
    seen = set()
    
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:  # avoid duplicates
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    
    return handles, labels
class ArgumentError(Exception):
     pass
def subplot_align(axis1, *axes, dim='x'):
    """
    Aligns the position of the given primary axis (axis1) with other specified axes based on the specified dimension.

    This function adjusts the position of the primary axis `axis1` to align with the other given axes along the specified dimension ('x', 'y', or 'both'). It listens for resize events on the figure canvas and dynamically adjusts the position of `axis1` to maintain alignment.

    Parameters:
    axis1 (matplotlib.axes.Axes): The primary axis whose position will be adjusted.
    *axes (matplotlib.axes.Axes): Additional axes to which the primary axis should be aligned.
    dim (str): The dimension along which to align the axes. Must be one of 'x', 'y', or 'both'. Default is 'x'.

    Raises:
    ArgumentError: If the specified dimension is not 'x', 'y', or 'both'.

    Returns:
    None

    Example:
    --------
    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    >>> subplot_align(ax1, ax2, ax3, dim='y')
    
    This will align `ax1` with `ax2` and `ax3` along the y-axis.
    """
    
    # Check if the specified dimension is valid
    if not dim.lower() in ['x', 'y', 'both']:
        raise ArgumentError(f'dimension to align is not understood. You chose: {dim}. Please specify either "x" or "y" or "both".')
    
    def onresize(axis1, axes, event):
        """
        Inner function to handle the resize event by adjusting the position of axis1.
        """
        if dim.lower() == 'x':
            # Concatenate all x-axis positions of the specified axes
            x = np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            # Calculate the new width and x-position for axis1
            width = max(x) - min(x)
            x = min(x)
            y = axis1.get_position().y0
            height = axis1.get_position().height
        elif dim.lower() == 'y':
            # Concatenate all y-axis positions of the specified axes
            y = np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            # Calculate the new height and y-position for axis1
            height = max(y) - min(y)
            y = min(y)
            x = axis1.get_position().x0
            width = axis1.get_position().width
        elif dim.lower() == 'both':
            # Concatenate all x and y positions of the specified axes
            x = np.concatenate([[ax.get_position().x0, ax.get_position().x1] for ax in axes])
            width = max(x) - min(x)
            x = min(x)
            y = np.concatenate([[ax.get_position().y0, ax.get_position().y1] for ax in axes])
            height = max(y) - min(y)
            y = min(y)
        # Set the new position of axis1
        return axis1.set_position([x, y, width, height])
    
    
def get_kde_sns_style(data, freqs, cut=3, gridsize=200, bw_adjust=1):
    from scipy.stats import gaussian_kde

    """
    Compute a KDE in the style of seaborn for given data.

    Parameters
    ----------
    data : np.ndarray
        Input data for KDE.
    cut : float, optional
        Extend the x-range past the data by this many bandwidths (default 3).
    gridsize : int, optional
        Number of points in the KDE grid (default 200).
    bw_adjust : float, optional
        Bandwidth adjustment factor (default 1).

    Returns
    -------
    xs : np.ndarray
        X values for KDE.'binary'
        The KDE object.
    """
    kde = gaussian_kde(data, bw_method='scott')
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)

    data_min, data_max = data.min(), data.max()
    data_range = data_max - data_min
    xmin = data_min - cut * data_range
    xmax = data_max + cut * data_range

    xs = np.linspace(xmin, xmax, gridsize)
    ys = kde(xs)
    return xs, ys, kde

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib as mpl
    import numpy as np
    if isinstance(cmap, str):
        cmap= mpl.colormaps[cmap]
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
def snap_to_nearest(values, support):
    """
    Snap each entry in `values` to the closest value in `support`.

    Parameters
    ----------
    values : array-like
        Input array of floats (random values).
    support : array-like
        Sorted or unsorted array of unique allowed values.

    Returns
    -------
    snapped : np.ndarray
        Array of same shape as `values`, where each element is replaced
        with the nearest value in `support`.
    """
    values = np.asarray(values)
    support = np.asarray(support)

    # ensure support is sorted
    support_sorted = np.sort(support)

    # indices of closest support values
    idxs = np.searchsorted(support_sorted, values, side="left")

    # clamp idxs into [0, len(support)-1]
    idxs = np.clip(idxs, 0, len(support_sorted)-1)

    # get candidate neighbors
    left_idx = np.maximum(idxs - 1, 0)
    right_idx = np.minimum(idxs, len(support_sorted)-1)

    # choose whichever neighbor is closer
    left_vals = support_sorted[left_idx]
    right_vals = support_sorted[right_idx]
    choose_right = np.abs(values - right_vals) < np.abs(values - left_vals)

    snapped = np.where(choose_right, right_vals, left_vals)
    return snapped


def mlt_ridge_plot(flims, var_col= 'fmin', Epoch_range=(-1.5, 1.5), Epoch_column='Epoch', MLT='MLT', fig=False, freqs = np.arange(20, 1041, 4),
               min_f=20, max_f=1040, cmap= truncate_colormap('binary', 0.3, 1),
               hist=False, bins = np.arange(0, 25, 2),
               figtitle= lambda t1, t2: f'Frequency Minimum Distribution\n Substorm Epoch {t1} to {t2} minutes',
               scale= .08/0.007,
               figtitle_size=50,
               whisker_x=.4,
               whisker_y= False,
               bw_adjust=0.5,
               **setup_fig_kwargs):
    def setup_figure(figsize=(13, 20), xlabel='Frequency [kHz]', min_f=20, max_f=1040,
                    bin_labels= [f"{int(b):02d}-{int(b+2):02d} MLT" for b in range(0, 24, 2)],
                    label_color='black', label_fontsize=20,
                    axis_label_size=35, offset_spacing= .083, trange=(-60, 30)):
            fig = plt.figure(figsize=(figsize))
            gs= fig.add_gridspec(1, 3, width_ratios=[.1, .95, .08], hspace=0, wspace=0.1)
            ax= fig.add_subplot(gs[1])
            label_ax= fig.add_subplot(gs[0], sharey=ax)
            label_ax.axis('off')
            label_ax.set_xlim(0, 1)
            numbers_ax= fig.add_subplot(gs[2], sharey=ax)
            numbers_ax.set_xlim(0, 1)
            numbers_ax.axis('off')
            numbers_ax.set_title('Number of Data Points', size=axis_label_size, y=.96, x=-.2)
            time_line_ax= fig.add_subplot([0, .05, 1, .04])
            time_line_ax.set_title('Epoch [min] Time Line', size=axis_label_size, y=.8)
            subplot_align(time_line_ax, label_ax, ax, numbers_ax, dim='x')
            time_line_ax.spines[['left', 'right', 'top',]].set_visible(False)
            time_line_ax.yaxis.set_visible(False)
            time_line_ax.spines['bottom'].set_position(('data', 0))
            time_line_ax.minorticks_on()
            time_line_ax.tick_params(axis='x', which='both',
                                    direction='inout', labelsize=label_fontsize)
            time_line_ax.tick_params(axis='x', which='major',
                                    size=25)
            time_line_ax.tick_params(axis='x', which='minor',
                                    size=7)
            time_line_ax.set_xticks(np.arange(trange[0], trange[-1]+5, 5))
            time_line_ax.set_xlim(*trange)
            time_line_ax.set_ylim(-1, 1)
            # time_line_ax.set_xlabel('Epoch\n[min]', x=1.09, rotation=270, labelpad=-90,
            #                         size=axis_label_size)
            # inv = time_line_ax.transAxes.inverted()
            for label, tick in zip(time_line_ax.get_xticklabels()[1::2],
                                time_line_ax.get_xticklines()[1::4]):
                path= tick.get_path()
                # y_coord = inv(tick.get_path().vertices[0, :])
                label.set_y(1.5)
                label.set_va('top')
            labels = [label.get_text() if i % 2 == 0 else '' for i, label in enumerate(time_line_ax.get_xticklabels())]
            time_line_ax.set_xticklabels(labels)
            ax.set_xticks(np.arange(min_f, max_f+100, 100))



            fig.subplots_adjust(bottom=0.2)
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax.set_xlim(min_f, max_f)
            fig.offsets= np.arange(0, (len(bin_labels))*offset_spacing + offset_spacing/2, offset_spacing)[-2::-1]
            ax.set_ylim(0, fig.offsets.max()+offset_spacing*1.1)
            ax.set_yticks(np.arange(fig.offsets.min(), fig.offsets.max()+offset_spacing, offset_spacing))
            ax.tick_params(axis='y', which='both', length=0, labelleft=False)
            # ax.set_ylim(0, 1)
            ax.set_xlabel(xlabel, size=axis_label_size, zorder=100)
            ax.tick_params(labelsize=label_fontsize)
            ax.minorticks_on()
            ax.grid(which='both', color='gray', linestyle=':', linewidth=2, axis='both')
            label_ax.mlt_labels= []
            numbers_ax.sub_nums= []
            
            for offset, mlt in zip(fig.offsets, bin_labels):
                label_ax.mlt_labels.append(label_ax.text(1, offset, mlt, ha='right', va='center',
                                                        color=label_color, fontsize=label_fontsize))

            scale_ax = fig.add_axes([ax.get_position().x1-0.1, ax.get_position().y0+.01, 0.01, ax.get_position().height])  # [left, bottom, width, height] in figure fraction

            scale_ax.sharey(ax)
            scale_ax.axis('off')
            scale_ax.whisker=[]
            time_line_ax.set_zorder(-100)
            return fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax
    def clear_figure(fig, main_ax, label_ax, numbers_ax, time_line_ax, scale_ax):
            for j in main_ax.collections: j.remove()
            for j in main_ax.lines: j.remove()
            for j in main_ax.patches: j.remove()
            for j in numbers_ax.sub_nums: j.remove()
            for j in time_line_ax.collections: j.remove()
            for j in scale_ax.whisker: j.remove()
            numbers_ax.sub_nums= []
            scale_ax.whisker= []
    import pandas as pd
    if isinstance(figtitle, str):
        base_title= figtitle
        figtitle = lambda t1, t2: f'{base_title}\n Epoch {t1} to {t2} minutes'
    if isinstance(cmap, str):
        import matplotlib as mpl
        cmap= mpl.colormaps[cmap]
    if any(bins%1):
        labels=[f"{b:04.1f}-{b+np.diff(bins)[0]:04.1f} MLT" for b in bins[:-1]]
    else:
        labels= [f"{int(b):02d}-{int(b)+int(np.diff(bins)[0]):02d} MLT" for b in bins[:-1]]
    if fig:
        fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax = fig
        clear_figure(fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax)
    else:
        fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax = setup_figure(min_f=min_f, max_f= max_f, bin_labels = labels, **setup_fig_kwargs)
    ind= (flims[Epoch_column] >= Epoch_range[0]) & (flims[Epoch_column] <= Epoch_range[-1])

    df_original = flims.loc[ind, [var_col, MLT]]
    df_original['MLT_bin'] = pd.cut(df_original[MLT], bins=bins, labels=labels, include_lowest=True, right=False)
    df = df_original[[var_col, 'MLT_bin']].dropna().copy()
    df.rename(columns={'MLT_bin': MLT}, inplace=True)
    mlt_groups = df.groupby(MLT, observed=False)
    # palette = cmap(len(df['MLT'].unique()))
    palette = cmap(np.abs(bins[:-1]+np.diff(bins)/2-12)/12)
    mlt_to_color = dict(zip(sorted(df[MLT].unique()), palette))
    
    whisker_height_ = setup_fig_kwargs.get('offset_spacing',.083)*.75
    freqs= np.array([  20,   24,   28,   32,   36,   40,   44,   48,   52,   60,   72,
                        80,   92,  104,  124,  136,  148,  176,  196,  224,  256,  272,
                        332,  388,  428,  484,  540,  624,  740,  804,  940, 1040])
    for offset, (mlt, group), zorder in zip(fig.offsets, mlt_groups,
                                np.arange(0, len(mlt_groups)*100, 100)):
        color = mlt_to_color[mlt]
        data = group[var_col].values
        data= data[np.isin(data, freqs)]
        while isinstance(data[-1], (np.ndarray, list)):
            data = np.concatenate(data)
        data = snap_to_nearest(data, freqs)

        numbers_ax.sub_nums.append(numbers_ax.text(0.5, offset+.001, len(group),
                                                   ha='center', va='bottom', size=setup_fig_kwargs.get('label_fontsize', 20),
                                                   color=setup_fig_kwargs.get('label_color', 'black')))
        xs, ys, kde = get_kde_sns_style(data, freqs, cut=3, gridsize=1000, bw_adjust=bw_adjust)
        # ys= (ys/ys.max())*scale
        ys= ys*scale

        if not cmap:
            ax.plot(xs, ys+offset, color='grey', lw=1.5, zorder=zorder+1)
        else:
            ax.plot(xs, ys+offset, color='white', lw=1.5, zorder=zorder+1)
            ax.fill_between(xs, ys+offset, y2=offset, color=color, alpha=.9, zorder=zorder)
        if hist:
            bin_width = 4  # since your bins are size 4
            # Compute histogram (bin counts)
            bin_edges = freqs - bin_width/2
            bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
            hist_counts, _ = np.histogram(data, bins=bin_edges)
            # Normalize so that sum(height * bin_width) = 1
            # hist_counts = hist_counts / (hist_counts.sum() * bin_width) if hist_counts.sum() > 0 else hist_counts
            hist_counts= (hist_counts/hist_counts.max())*1/len(bins)
            # Optionally scale for visual clarity (e.g., to match KDE peak height)
            hist_counts = hist_counts * (ys.max()*.9/hist_counts.max())
            # Plot histogram as bars under the ridgeline
            ax.bar(freqs, hist_counts, width=bin_width, bottom=offset, align='center',
                color='grey', alpha=0.9, edgecolor='white', linewidth=0.5, zorder=zorder+2)
    whisker_val = round(whisker_height_/scale, 3)
    whisker_height = whisker_val*scale
    if whisker_y:
        whisker_y = fig.offsets[int(len(fig.offsets)/2)]
    scale_ax.whisker.extend(scale_ax.plot([0.5, 0.5], [whisker_y, whisker_height+whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.extend(scale_ax.plot([0.3, 0.7], [whisker_y, whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.extend(scale_ax.plot([0.3, 0.7], [whisker_height+whisker_y, whisker_height+whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.append(scale_ax.text(1.0, whisker_height / 2 +whisker_y, f'{whisker_val}\nPDE', va='center', fontsize=setup_fig_kwargs.get('label_fontsize', 20)))
    time_line_ax.fill_betweenx([-10, 10], *Epoch_range, color='grey', zorder=-100)
    fig.suptitle(figtitle(*Epoch_range), size=figtitle_size)
    return fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax

def time_ridge_plot(flims, var_col= 'fmin', MLT_range= (10, 14), Epoch_column='Epoch', MLT='MLT', fig=False, freqs = np.arange(20, 1041, 4),
               min_f=20, max_f=1040, cmap= truncate_colormap('binary', 0.3, 1),
               hist=False, Epochs= [-30, -183/(2*60), 183/(2*60), 60, 120],
               figtitle= f'Frequency Minimum Distribution',
               scale= .08/0.007,
               figtitle_size=50,
               whisker_x=.4,
               whisker_y= False,
               bw_adjust=0.5,
               trange=(-60, 130),
               **setup_fig_kwargs):
    def epoch2label(Epochs):
        """
        Convert epoch tuples to formatted labels.

        Parameters
        ----------
        Epochs : list of tuples
            List of tuples representing epochs.

        Returns
        -------
        list of str
            Formatted labels for each epoch.
        """
        return [f'{t1+(t2-t1)/2} Min' for (t1, t2) in Epochs]
    Epochs= [(E-183/(2*60), E+183/(2*60)) for E in Epochs]
    Epochs_labels= epoch2label(Epochs)
    Epoch_mids= np.array([Epoch[0]+np.diff(Epoch)/2 for Epoch in Epochs])
    Epoch_mids+=np.append(Epoch_mids[Epoch_mids<0], 0).min()
    palette = cmap(Epoch_mids/Epoch_mids.max())
    palette = cmap(np.linspace(0, 1, len(Epochs_labels)))
    epoch_to_color = dict(zip(Epochs_labels, palette))
    import pandas as pd
    if isinstance(figtitle, str):
        base_title= figtitle
        figtitle = lambda mlt1, mlt2: f'{base_title}\n {mlt1} to {mlt2} MLT'


    def setup_figure(figsize=(13, 20), xlabel='Frequency [kHz]', min_f=20, max_f=1040,
                     offset_spacing= .081, axis_label_size=35, label_fontsize=30, label_color='black'):
        fig = plt.figure(figsize=(figsize))
        gs= fig.add_gridspec(1, 3, width_ratios=[.1, .95, .08], hspace=0, wspace=0.1)
        ax= fig.add_subplot(gs[1])
        label_ax= fig.add_subplot(gs[0], sharey=ax)
        label_ax.axis('off')
        label_ax.set_xlim(0, 1)
        numbers_ax= fig.add_subplot(gs[2], sharey=ax)
        numbers_ax.set_xlim(0, 1)
        numbers_ax.axis('off')
        numbers_ax.set_title('Number of Data Points', size=axis_label_size, y=.96, x=-.2)
        time_line_ax= fig.add_subplot([0, .05, 1, .04])
        time_line_ax.set_title('Epoch [min] Time Line', size=axis_label_size, y=.8)
        subplot_align(time_line_ax, label_ax, ax, numbers_ax, dim='x')
        time_line_ax.spines[['left', 'right', 'top',]].set_visible(False)
        time_line_ax.yaxis.set_visible(False)
        time_line_ax.spines['bottom'].set_position(('data', 0))
        time_line_ax.minorticks_on()
        time_line_ax.tick_params(axis='x', which='both',
                                    direction='inout', labelsize=label_fontsize)
        time_line_ax.tick_params(axis='x', which='major',
                                    size=25)
        time_line_ax.tick_params(axis='x', which='minor',
                                    size=7)
        time_line_ax.set_xticks(np.arange(trange[0], trange[-1]+5, 10))
        time_line_ax.set_xlim(*trange)
        time_line_ax.set_ylim(-1, 1)
        # time_line_ax.set_xlabel('Epoch\n[min]', x=1.08, rotation=270, labelpad=-85,
        #                         size=axis_label_size)
        # inv = time_line_ax.transAxes.inverted()
        for label, tick in zip(time_line_ax.get_xticklabels()[1::2],
                                time_line_ax.get_xticklines()[1::4]):
            path= tick.get_path()
            # y_coord = inv(tick.get_path().vertices[0, :])
            label.set_y(1.8)
            label.set_va('top')
        labels = [label.get_text() if i % 2 == 0 else '' for i, label in enumerate(time_line_ax.get_xticklabels())]
        time_line_ax.set_xticklabels(labels)


        fig.subplots_adjust(bottom=0.2)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_xlim(min_f, max_f)
        # ax.set_yticks([])
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, size=axis_label_size)
        ax.tick_params(labelsize=label_fontsize)
        # ax.grid()
        ax.minorticks_on()

        ax.grid(which='both', color='gray', linestyle=':', linewidth=2, axis='both')

        # Hide minor tick marks on spines (i.e., no little ticks on axes)
        # ax.tick_params(axis='both', which='minor', length=0)
        label_ax.epoch_labels= []
        numbers_ax.sub_nums= []
        fig.offsets= np.arange(0, (len(Epochs_labels))*offset_spacing + offset_spacing/2, offset_spacing)[-2::-1]
        ax.set_ylim(0, fig.offsets.max()+offset_spacing)
        scale_ax = fig.add_axes([ax.get_position().x1-0.1, ax.get_position().y0+.01, 0.01, ax.get_position().height])  # [left, bottom, width, height] in figure fraction
        for offset, Epoch, label in zip(fig.offsets, Epochs, Epochs_labels):
            hatch= None
            label_ax.epoch_labels.append(label_ax.text(1, offset, label, ha='right', va='center',
                                                        color=label_color, fontsize=label_fontsize))
            color = epoch_to_color[label]
            time_line_ax.fill_betweenx([-.6, 10], *Epoch, color=color, hatch=hatch, edgecolor='white')
        scale_ax.sharey(ax)
        scale_ax.axis('off')
        scale_ax.whisker=[]
        return fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax
    def clear_figure(fig, main_ax, label_ax, numbers_ax, time_line_ax, scale_ax):
            for j in main_ax.collections: j.remove()
            for j in main_ax.lines: j.remove()
            for j in main_ax.patches: j.remove()
            for j in numbers_ax.sub_nums: j.remove()
            for j in time_line_ax.collections: j.remove()
            for j in scale_ax.whisker: j.remove()
            numbers_ax.sub_nums= []
            scale_ax.whisker= []
    if fig:
        fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax = fig
        clear_figure(fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax)
    else:
        fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax = setup_figure(min_f=min_f, max_f= max_f, **setup_fig_kwargs)
    ind= (flims[MLT] >= MLT_range[0]) & (flims[MLT] <= MLT_range[-1])
    df_original = flims.loc[ind, [var_col, Epoch_column]]
    df_original['Epoch_bin'] = pd.cut(df_original[Epoch_column], bins=np.unique(np.array([Epochs]).flatten()), 
                                    labels=epoch2label(zip(np.unique(np.array([Epochs]).flatten())[:-1],
                                                            np.unique(np.array([Epochs]).flatten())[1:])),
                                                        include_lowest=True, right=False)
    df_original= df_original[df_original.Epoch_bin.isin(Epochs_labels)]
    df = df_original[[var_col, 'Epoch_bin']].dropna().copy()
    df.rename(columns={'Epoch_bin': Epoch_column}, inplace=True)
    epoch_groups = df.groupby(Epoch_column, observed=True)
        
    whisker_height_ = setup_fig_kwargs.get('offset_spacing',.081)*.75
    scale= .08/0.007
    for offset, (epoch, group), zorder in zip(fig.offsets, epoch_groups,
                                np.arange(0, len(epoch_groups)*100, 100)):
        hatch= None
        color = epoch_to_color[epoch]
        data = group[var_col].values
        while isinstance(data[-1], (np.ndarray, list)):
            data = np.concatenate(data)
        numbers_ax.sub_nums.append(numbers_ax.text(0.5, offset+.001, len(group),
                                                    ha='center', va='bottom', size=setup_fig_kwargs.get('label_fontsize', 30),
                                                    color=setup_fig_kwargs.get('label_color', 'black')))
        xs, ys, kde = get_kde_sns_style(data, cut=3, gridsize=1000, bw_adjust=0.5, freqs=freqs)
        # Draw whisker
        # whisker_val = round(whisker_height_*ys.max()/scale, 3)
        # whisker_height = (whisker_val/ys.max())*scale

        # ys= (ys/ys.max())*scale
        ys= ys*scale

        if not cmap:
            ax.plot(xs, ys+offset, color='grey', lw=1.5, zorder=zorder+1)
        else:
            ax.plot(xs, ys+offset, color='white', lw=1.5, zorder=zorder+1)
            ax.fill_between(xs, ys+offset, y2=offset, color=color, alpha=.9, zorder=zorder, hatch=hatch, edgecolor='white')
        if hist:
            bin_width = 4  # since your bins are size 4
            # Compute histogram (bin counts)
            bin_edges = freqs - bin_width/2
            bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
            hist_counts, _ = np.histogram(data, bins=bin_edges)
            # Normalize so that sum(height * bin_width) = 1
            # hist_counts = hist_counts / (hist_counts.sum() * bin_width) if hist_counts.sum() > 0 else hist_counts
            hist_counts= (hist_counts/hist_counts.max())*1/len(Epochs)
            # Optionally scale for visual clarity (e.g., to match KDE peak height)
            hist_counts = hist_counts * (ys.max()*.9/hist_counts.max())
            # Plot histogram as bars under the ridgeline
            ax.bar(freqs, hist_counts, width=bin_width, bottom=offset, align='center',
                color='grey', alpha=0.9, edgecolor='white', linewidth=0.5, zorder=zorder+2)


    whisker_val = round(whisker_height_/scale, 3)
    whisker_height = whisker_val*scale
    if whisker_y:
        whisker_y = fig.offsets[int(len(fig.offsets)/2)]
    scale_ax.whisker.extend(scale_ax.plot([0.5, 0.5], [whisker_y, whisker_height+whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.extend(scale_ax.plot([0.3, 0.7], [whisker_y, whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.extend(scale_ax.plot([0.3, 0.7], [whisker_height+whisker_y, whisker_height+whisker_y], 'k', linewidth=1.5))
    scale_ax.whisker.append(scale_ax.text(1.0, whisker_height / 2 +whisker_y, f'{whisker_val}\nPDE', va='center', fontsize=setup_fig_kwargs.get('label_fontsize', 20)))
    fig.suptitle(figtitle(*MLT_range), size=figtitle_size)
    return fig, ax, label_ax, numbers_ax, time_line_ax, scale_ax
