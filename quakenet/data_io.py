"""Handle the raw data input/output and interface with external formats."""

from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
import datetime as dt


def load_stream(path):
    """Loads a Stream object from the file at path.

    Args:
        path: path to the input file, (for supported formats see,
        http://docs.obspy.org/tutorial/code_snippets/reading_seismograms.html)

    Returns:
        an obspy.core.Stream object
        (http://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html#obspy.core.stream.Stream)
    """

    stream = read(path)
    stream.merge()

    # assert len(stream) == 3  # We need X,Y,Z traces

    return stream


def load_catalog(path):
    """Loads a event catalog from a .csv file.

    Each row in the catalog references a know seismic event.

    Args:
        path: path to the input .csv file.

    Returns:
        catalog: A Pandas dataframe.
    """

    catalog = pd.read_csv(path)
    # Check if utc_timestamp exists, otherwise create it
    if 'utc_timestamp' not in catalog.columns:
        utc_timestamp = []
        for e in catalog.origintime.values:
            utc_timestamp.append(UTCDateTime(e).timestamp)
        catalog['utc_timestamp'] = utc_timestamp
    return catalog


def write_stream(stream, path):
    stream.write(path, format='MSEED')


def write_catalog(events, path):
    catalog = pd.DataFrame(
        {'utc_timestamp': pd.Series([t.timestamp for t in events])})
    catalog.to_csv(path)

def write_catalog_with_clusters(events, clusters, latitudes, longitudes, depths, path):
    catalog = pd.DataFrame(
        {'utc_timestamp': pd.Series([t for t in events]),
         "cluster_id": pd.Series([cluster_id for cluster_id in clusters]),
         "latitude": pd.Series([lat for lat in latitudes]),
         "longitude": pd.Series([lon for lon in longitudes]),
         "depth": pd.Series([d for d in depths])})
    catalog.to_csv(path)

