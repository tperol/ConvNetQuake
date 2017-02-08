import numpy as np
import obspy.core.trace as trace
import obspy.core.stream as stream
import pandas as pd

from obspy.core.utcdatetime import UTCDateTime

def stream2array(stream):
    nchan = len(stream)
    if nchan < 1:
        raise ValueError('empty stream')
    npts = stream[0].stats.npts
    out = np.zeros((nchan, npts), dtype=stream[0].data.dtype)
    for c in range(nchan):
        if stream[c].stats.npts != npts:
            raise ValueError('stream has traces of different lengths')
        out[c, :] = stream[c].data
    return out


def array2stream(array, sampling_rate, stream_info):
    nchans = array.shape[0]
    npts = array.shape[1]
    traces = []
    for c in range(nchans):
        traces.append(trace.Trace(data=array[c, :],
                                  header={'sampling_rate': sampling_rate,
                                          'station': stream_info['station'],
                                          'network': stream_info['network'],
                                          'channel': stream_info['channels'][c]}))
    return stream.Stream(traces)


def convert_catalog(src, dst, src_format='OK'):
    """Convert external catalogs to our custom catalog format.

    Different adapters for different src formats.

    Columns:
        utc_timestamp: (float) use with obspy UTCDateTime(t).
        label: label of a unique source that caused the event.
    """

    src_cat = pd.read_csv(src)

    # - Adapters -----------------------------
    if src_format == 'OK':
        count = src_cat.shape[0]
        utc_timestamp = [UTCDateTime(t).timestamp for t in src_cat['origintime']]
        label = np.zeros(count, dtype=int)
        latitude = src_cat['latitude']
        longitude = src_cat['longitude']
        depth = src_cat['depth']            # in km
        err_lat = src_cat['err_lat']        # in km
        err_lon = src_cat['err_lon']        # in km
        err_depth = src_cat['err_depth']    # in km
    else:
        raise ValueError('unknown catalog format "{}"'.format(src_format))
    # ----------------------------------------

    df = pd.DataFrame({
        'utc_timestamp': utc_timestamp,
        'label': label,
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        'err_lat': err_lat,
        'err_lon': err_lon,
        'err_depth': err_depth})
    df.to_csv(dst)

    print 'Converted catalog with {} entries.'.format(count)
