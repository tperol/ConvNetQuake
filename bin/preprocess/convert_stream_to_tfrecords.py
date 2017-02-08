#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : convert_stream_to_tfrecords.py
# Creation Date : 09-12-2016
# Last Modified : Mon Jan  9 13:21:32 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
#TODO: Generating windows is embarassingly parallel. This can be speed up
""""
Load a stream, preprocess it, and create windows in tfrecords to be
fed to ConvNetQuake for prediction
NB: Use max_windows to create tfrecords of 1 week or 1 month
NB2: This convert a stream into ONE tfrecords, this is different
from create_dataset_events and create_dataset_noise that output
multiple tfrecords of equal size used for training.
e.g.,
./bin/preprocess/convert_stream_to_tfrecords.py \
--stream_path data/streams/GSOK029_7-2014.mseed \
--output_dir  data/tfrecord \
--window_size 10 --window_step 11 \
--max_windows 5000
"""
import os
import setproctitle
import numpy as np
from quakenet.data_pipeline import DataWriter
import tensorflow as tf
from obspy.core import read
from quakenet.data_io import load_catalog
from obspy.core.utcdatetime import UTCDateTime
import fnmatch
import json
import argparse
from tqdm import tqdm
import time
import pandas as pd

def preprocess_stream(stream):
    stream = stream.detrend('constant')
    return stream.normalize()

def write_json(metadata,output_metadata):
    with open(output_metadata, 'w') as outfile:
        json.dump(metadata, outfile)

def main(args):
    setproctitle.setproctitle('quakenet_predict_from_tfrecords')

    # Create dir to store tfrecords
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load stream
    stream_path = args.stream_path
    stream_file = os.path.split(stream_path)[-1]
    print "+ Loading Stream {}".format(stream_file)
    stream = read(stream_path)
    print '+ Preprocessing stream'
    stream = preprocess_stream(stream)

    # Dictionary of nb of events per tfrecords
    metadata = {}
    output_metadata = os.path.join(args.output_dir,"metadata.json")

    # Csv of start and end times
    times_csv = {}
    times_csv = {"start_time": [],
                 "end_time": []}

    # Write event waveforms and cluster_id=-1 in .tfrecords
    output_name = stream_file.split(".mseed")[0] + ".tfrecords"
    output_path = os.path.join(args.output_dir, output_name)
    writer = DataWriter(output_path)

    # Create window generator
    win_gen = stream.slide(window_length=args.window_size,
                           step=args.window_step,
                           include_partial_windows=False)
    if args.max_windows is None:
        total_time = stream[-1].stats.endtime - stream[0].stats.starttime
        max_windows = (total_time - args.window_size) / args.window_step
        print "total time {}, wind_size {}, win_step {}".format(
              total_time, args.window_size, args.window_step)
    else:
        max_windows = args.max_windows

    start_time = time.time()
    for idx, win in tqdm(enumerate(win_gen),total=int(max_windows),
                         unit="window",leave=False):

        # If there is not trace skip this waveform
        n_traces = len(win)
        if n_traces == 0:
            continue
        # Check trace is complete
        if len(win)==3:
            n_samples = min(len(win[0].data),len(win[1].data))
            n_samples = min(n_samples, len(win[2].data))
        else:
            n_sample = 10
        n_pts = win[0].stats.sampling_rate * args.window_size + 1
        # there is no event
        if (len(win)==3) and (n_pts == n_samples):
            # Write tfrecords
            writer.write(win,-1)
            # Write start and end times in csv
            times_csv["start_time"].append(win[0].stats.starttime)
            times_csv["end_time"].append(win[0].stats.endtime)
            # Plot events
            if args.plot:
                trace = win[0]
                viz_dir = os.path.join(
                    args.output_dir, "viz", stream_file.split(".mseed")[0])
                if not os.path.exists(viz_dir):
                    os.makedirs(viz_dir)
                trace.plot(outfile=os.path.join(viz_dir,
                                                "window_{}.png".format(idx)))

        # if idx % 1000  ==0 and idx != 0:
        #     print "{} windows created".format(idx)

        if idx == max_windows:
            break

    # Cleanup writer
    print("Number of windows  written={}".format(writer._written))
    writer.close()

    # Write metadata
    metadata[stream_file.split(".mseed")[0]] = writer._written
    write_json(metadata, output_metadata)

    # Write start and end times
    df = pd.DataFrame.from_dict(times_csv)
    output_times = os.path.join(args.output_dir,"catalog_times.csv")
    df.to_csv(output_times)

    print "Last window analyzed ends on", win[0].stats.endtime
    print "Time to create tfrecords: {}s".format(time.time()-start_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,default=None,
                        help="path to mseed to analyze")
    parser.add_argument("--window_size",type=int,default=10,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=11,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to create")
    parser.add_argument("--output_dir",type=str,default="output/predict",
                        help="dir of predicted events")
    parser.add_argument("--plot",type=bool,default=False,
                        help="pass this flag to plot windows")
    args = parser.parse_args()

    main(args)
