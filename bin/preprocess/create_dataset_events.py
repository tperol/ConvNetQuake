#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : create_dataset_events.py
# Creation Date : 05-12-2016
# Last Modified : Fri Jan  6 15:04:54 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""Creates tfrecords dataset of events trace and their cluster_ids.
This is done by loading a dir of .mseed and one catalog with the
time stamps of the events and their cluster_id
e.g.,
./bin/preprocess/create_dataset_events.py \
--stream_dir data/streams \
--catalog data/50_clusters/catalog_with_cluster_ids.csv\
--output_dir data/50_clusters/tfrecords
"""

import os
import numpy as np
from quakenet.data_pipeline import DataWriter
import tensorflow as tf
from obspy.core import read
from quakenet.data_io import load_catalog
from obspy.core.utcdatetime import UTCDateTime
from openquake.hazardlib.geo.geodetic import distance
import fnmatch
import json

flags = tf.flags
flags.DEFINE_string('stream_dir', None,
                    'path to the directory of streams to preprocess.')
flags.DEFINE_string(
    'catalog', None, 'path to the events catalog to use as labels.')
flags.DEFINE_string('output_dir', None,
                    'path to the directory in which the tfrecords are saved')
flags.DEFINE_bool("plot", False,
                  "If we want the event traces to be plotted")
flags.DEFINE_float(
    'window_size', 10, 'size of the window samples (in seconds)')
flags.DEFINE_float('v_mean', 5.0, 'mean velocity')
flags.DEFINE_boolean("save_mseed",False,
                     "save the windows in mseed format")
FLAGS = flags.FLAGS


def distance_to_station(lat, long, depth):
    # station GPS coordinates
    lat0 = 35.796570
    long0 = -97.454860
    depth0 = -0.333
    # return distance of the event to the station
    return distance(long, lat, depth, long0, lat0, depth0)


def preprocess_stream(stream):
    stream = stream.detrend('constant')
    return stream.normalize()


def filter_catalog(cat):
    # Filter around Guthrie sequence
    cat = cat[(cat.latitude > 35.7) & (cat.latitude < 36)
              & (cat.longitude > -97.6) & (cat.longitude < -97.2)]
    return cat


def get_travel_time(catalog):
    """Find the time between origin and propagation"""
    v_mean = FLAGS.v_mean
    coordinates = [(lat, lon, depth) for (lat, lon, depth) in zip(catalog.latitude,
                                                                  catalog.longitude,
                                                                  catalog.depth)]
    distances_to_station = [distance_to_station(lat, lon, depth)
                            for (lat, lon, depth) in coordinates]
    travel_time = [distance/v_mean for distance in distances_to_station]
    return travel_time

def write_json(metadata,output_metadata):
    with open(output_metadata, 'w') as outfile:
        json.dump(metadata, outfile)


def main(_):

    stream_files = [file for file in os.listdir(FLAGS.stream_dir) if
                    fnmatch.fnmatch(file, '*.mseed')]
    print "List of streams to anlayze", stream_files

    # Create dir to store tfrecords
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Dictionary of nb of events per tfrecords
    metadata = {}
    output_metadata = os.path.join(FLAGS.output_dir,"metadata.json")

    # Load Catalog
    print "+ Loading Catalog"
    cat = load_catalog(FLAGS.catalog)
    cat = filter_catalog(cat)

    for stream_file in stream_files:

        # Load stream
        stream_path = os.path.join(FLAGS.stream_dir, stream_file)
        print "+ Loading Stream {}".format(stream_file)
        stream = read(stream_path)
        print '+ Preprocessing stream'
        stream = preprocess_stream(stream)

        # Filter catalog according to the loaded stream
        start_date = stream[0].stats.starttime
        end_date = stream[-1].stats.endtime
        print("-- Start Date={}, End Date={}".format(start_date, end_date))

        filtered_catalog = cat[
            ((cat.utc_timestamp >= start_date)
             & (cat.utc_timestamp < end_date))]

        # Propagation time from source to station
        travel_time = get_travel_time(filtered_catalog)

        # Write event waveforms and cluster_id in .tfrecords
        output_name = stream_file.split(".mseed")[0] + ".tfrecords"
        output_path = os.path.join(FLAGS.output_dir, output_name)
        writer = DataWriter(output_path)
        print("+ Creating tfrecords for {} events".format(filtered_catalog.shape[0]))
        # Loop over all events in the considered stream
        for event_n in range(filtered_catalog.shape[0]):
            event_time = filtered_catalog.utc_timestamp.values[event_n]
            event_time += travel_time[event_n]
            st_event = stream.slice(UTCDateTime(event_time),
                                    UTCDateTime(event_time) + FLAGS.window_size).copy()
            cluster_id = filtered_catalog.cluster_id.values[event_n]
            n_traces = len(st_event)
            # If there is not trace skip this waveform
            if n_traces == 0:
                continue
            n_samples = len(st_event[0].data)
            n_pts = st_event[0].stats.sampling_rate * FLAGS.window_size + 1
            if (len(st_event) == 3) and (n_pts == n_samples):
                # Write tfrecords
                writer.write(st_event, cluster_id)
                # Save window and cluster_id
                if FLAGS.save_mseed:
                    output_label = "label_{}_lat_{:.3f}_lon_{:.3f}.mseed".format(
                                    cluster_id,
                                    filtered_catalog.latitude.values[event_n],
                                    filtered_catalog.longitude.values[event_n])
                    output_mseed_dir = os.path.join(FLAGS.output_dir,"mseed")
                    if not os.path.exists(output_mseed_dir):
                        os.makedirs(output_mseed_dir)
                    output_mseed = os.path.join(output_mseed_dir,output_label)
                    st_event.write(output_mseed,format="MSEED")
                # Plot events
                if FLAGS.plot:
                    trace = st_event[0]
                    viz_dir = os.path.join(
                        FLAGS.output_dir, "viz", stream_file.split(".mseed")[0])
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    trace.plot(outfile=os.path.join(viz_dir,
                                                    "event_{}.png".format(event_n)))
            else:
                print "Missing waveform for event:", UTCDateTime(event_time)

        # Cleanup writer
        print("Number of events written={}".format(writer._written))
        writer.close()
        # Write metadata
        metadata[stream_file.split(".mseed")[0]] = writer._written
        write_json(metadata, output_metadata)


if __name__ == "__main__":
    tf.app.run()
