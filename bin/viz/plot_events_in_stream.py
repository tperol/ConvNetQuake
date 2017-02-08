#!/usr/bin/env python # -------------------------------------------------------------------
# File Name : plot_events_in_stream.py
# Creation Date : 04-12-2016
# Last Modified : Wed Dec  7 17:25:54 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""
Load a catalog and a stream, plot the traces of the events
e.g.,
./bin/viz/plot_events
--catalog data/hackathon/catalog/catalog_with_cluster_ids.csv
--stream data/streams/GSOK029_12-2014.mseed
--output data/hackathon/2014_events/december_events
"""

from obspy.core import read
from quakenet.data_io import load_catalog
from obspy.core.utcdatetime import UTCDateTime
from openquake.hazardlib.geo.geodetic import distance
import numpy as np
import tensorflow as tf
import os

flags = tf.flags
flags.DEFINE_string('catalog',
                    None, 'path to the catalog')
flags.DEFINE_string('stream',
                    None, 'path to the stream')
flags.DEFINE_string('output',
                    None, 'path to the stream')
flags.DEFINE_float('v_mean',
                   5.0, 'mean velocity')
flags.DEFINE_integer('window_size',10,'size of the plotted window')
flags.DEFINE_boolean("with_preprocessing",False,
                     "pass this flag to plot after preprocessing")
FLAGS = flags.FLAGS


def distance_to_station(lat, long, depth):
    # station GPS coordinates
    lat0 = 35.796570
    long0 = -97.454860
    depth0 = -0.333
    # return distance of the event to the station
    return distance(long, lat, depth, long0, lat0, depth0)


def filter_catalog(cat):
    # Filter around Guthrie sequence
    cat = cat[(cat.latitude > 35.7) & (cat.latitude < 36)
              & (cat.longitude > -97.6) & (cat.longitude < -97.2)]
    return cat

def preprocess_stream(stream):
    stream = stream.detrend('constant')
    return stream.normalize()

def get_travel_time(catalog):
    """Find the time between origin and propagation"""
    v_mean = FLAGS.v_mean
    coordinates = [(lat, lon, depth)
                   for (lat, lon, depth) in zip(catalog.latitude,
                                                catalog.longitude,
                                                catalog.depth)]
    distances_to_station = [distance_to_station(lat, lon, depth)
                            for (lat, lon, depth) in coordinates]
    travel_time = [distance/v_mean for distance in distances_to_station]
    return travel_time

def main(_):

    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    # Load Catalog
    cat_path = FLAGS.catalog
    cat = load_catalog(cat_path)
    cat = filter_catalog(cat)

    # Load stream
    stream_path = FLAGS.stream
    print " + Loading stream"
    stream = read(stream_path)
    if FLAGS.with_preprocessing:
        print " + Preprocessing stream"
        stream = preprocess_stream(stream)

    # Filter catalog according to the loaded stream
    start_date = stream[0].stats.starttime
    end_date = stream[-1].stats.endtime
    print(" + Loaded Stream with Start Date={} and End Date={}".format(start_date, end_date))

    filtered_catalog = cat[
        ((cat.utc_timestamp >= start_date)
         & (cat.utc_timestamp < end_date))]

    travel_time = get_travel_time(filtered_catalog)

    print(" + Plotting {} events".format(filtered_catalog.shape[0]))
    for event_n in range(filtered_catalog.shape[0]):
        event_time = filtered_catalog.utc_timestamp.values[event_n] + travel_time[event_n]
        cluster_id= filtered_catalog.cluster_id.values[event_n]
        st_event = stream.slice(
            UTCDateTime(event_time), UTCDateTime(event_time) +
                        FLAGS.window_size).copy()
        if len(st_event)==3:
            trace = st_event[0]
            name_png = "event_{}_clusterid_{}.png".format(event_n,cluster_id)
            trace.plot(outfile=os.path.join(FLAGS.output,name_png))
        else:
            print st_event


if __name__ == "__main__":
    tf.app.run()
