#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : windows_from_detection_catalog.py
# Creation Date : 08-01-2017
# Last Modified : Sat Jan 21 16:48:57 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Load a catalog of detected events created by either
predict_from_stream.py or predict_from_tfrecords.py.
Pass --plot to plot the windows of found events.
Pass --save_sac to save the windows into sac files
"""

from quakenet.data_io import load_catalog, load_stream
import os
import argparse
import shutil
from obspy.core.utcdatetime import UTCDateTime
import numpy as np
from tqdm import tqdm

def main(args):
    # Remove previous output directory
    output_viz = os.path.join(args.output,"viz")
    output_sac = os.path.join(args.output,"sac")
    if args.plot:
        if os.path.exists(output_viz):
            shutil.rmtree(output_viz)
        os.makedirs(output_viz)
    if args.save_sac:
        if os.path.exists(output_sac):
            shutil.rmtree(output_sac)
        os.makedirs(output_sac)

    # Read stream
    print "+ Loading stream"
    st = load_stream(args.stream_path)
    # Read catalog
    print "+ Loading catalog"
    cat = load_catalog(args.catalog_path)

    # Look events in catalog and plot windows
    print "+ Creating windows with detected events from ConvNetQuake"
    for event in tqdm(range(cat.shape[0]),total=cat.shape[0],
                      unit="events",leave=False):
        win_start = UTCDateTime(cat.iloc[event].start_time)
        win_end = UTCDateTime(cat.iloc[event].end_time)
        win = st.slice(win_start, win_end).copy()
        if args.plot:
            win.plot(outfile=os.path.join(output_viz,
                        "event_{}.png".format(event)))
        if args.save_sac:
            for tr in win:
                    if isinstance(tr.data, np.ma.masked_array):
                                tr.data = tr.data.filled()
            win.write(os.path.join(output_sac,
                        "event_{}_.sac".format(event)), format="SAC")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,default=None,
                        help="path to mseed to analyze")
    parser.add_argument("--catalog_path",type=str,default=None,
                        help="path to directory of chekpoints")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to plot")
    parser.add_argument("--output",type=str,default="output/predict",
                        help="dir of predicted events")
    parser.add_argument("--plot",action="store_true",
                        help="pass this flag to plot")
    parser.add_argument("--save_sac",action="store_true",
                        help="pass flag to save windows of events in sac files")
    args = parser.parse_args()

    main(args)
