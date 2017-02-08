#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : predict_from_tfrecords.py
# Creation Date : 09-12-2016
# Last Modified : Mon Jan  9 16:04:01 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Prediction from a tfrecords. Create a catalog of found events
with their cluster id, cluster proba, start event time of the window, end
event time of the window
"""

import os
import setproctitle
import time
import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime


import quakenet.models as models
from quakenet.data_pipeline import DataPipeline
import quakenet.config as config

def main(args):
    setproctitle.setproctitle('quakenet_eval')

    if args.n_clusters == None:
        raise ValueError('Define the number of clusters with --n_clusters')

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_clusters = args.n_clusters
    cfg.add = 1
    cfg.n_clusters += 1
    cfg.n_epochs = 1

    # Remove previous output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    if args.plot:
        os.makedirs(os.path.join(args.output,"viz"))

    # data pipeline
    data_pipeline = DataPipeline(args.dataset, config=cfg,
                                    is_training=False)
    samples = {
        'data': data_pipeline.samples,
        'cluster_id': data_pipeline.labels,
        'start_time': data_pipeline.start_time,
        'end_time': data_pipeline.end_time}

    # set up model and validation metrics
    model = models.get(args.model, samples, cfg,
                        args.checkpoint_dir,
                        is_training=False)

    if args.max_windows is None:
        max_windows = 2**31
    else:
        max_windows = args.max_windows

    # Dictonary to store info on detected events
    events_dic ={"start_time": [],
                 "end_time": [],
                 "utc_timestamp": [],
                 "cluster_id": [],
                 "clusters_prob": []}

    # Create catalog name in which the events are stored
    output_catalog = os.path.join(args.output,'catalog_detection.csv')
    print 'Catalog created to store events', output_catalog


    # Run ConvNetQuake
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        model.load(sess,args.step)
        print 'Predicting using model at step {}'.format(
                sess.run(model.global_step))

        step = tf.train.global_step(sess, model.global_step)

        n_events = 0
        idx = 0
        time_start = time.time()
        while True:
            try:
                # Fetch class_proba and label
                to_fetch = [samples['data'],
                            model.layers['class_prob'],
                            model.layers['class_prediction'],
                            samples['start_time'],
                            samples['end_time']]
                sample, class_prob_, cluster_id, start_time, end_time = sess.run(to_fetch)

                # # Keep only clusters proba, remove noise proba
                clusters_prob = class_prob_[0,1::]
                cluster_id -= 1

                # label for noise = -1, label for cluster \in {0:n_clusters}

                is_event = cluster_id[0] > -1
                if is_event:
                    n_events += 1

                idx +=1
                if idx % 1000 ==0:
                    print "processed {} windows".format(idx)

                if is_event:
                    events_dic["start_time"].append(UTCDateTime(start_time))
                    events_dic["end_time"].append(UTCDateTime(end_time))
                    events_dic["utc_timestamp"].append((start_time +
                                                        end_time)/2.0)
                    events_dic["cluster_id"].append(cluster_id[0])
                    events_dic["clusters_prob"].append(list(clusters_prob))

                if idx >= max_windows:
                    print "stopped after {} windows".format(max_windows)
                    print "found {} events".format(n_events)
                    break

            except KeyboardInterrupt:
                print "processed {} windows, found {} events".format(idx+1,n_events)
                print "Run time: ", time.time() - time_start

            except tf.errors.OutOfRangeError:
                print 'Evaluation completed ({} epochs).'.format(cfg.n_epochs)
                break

        print 'joining data threads'
        m, s = divmod(time.time() - time_start, 60)
        print "Prediction took {} min {} seconds".format(m,s)
        coord.request_stop()
        coord.join(threads)


    # Dump dictionary into csv file
    df = pd.DataFrame.from_dict(events_dic)
    df.to_csv(output_catalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None,
                        help="path to tfrecords to analyze")
    parser.add_argument("--checkpoint_dir",type=str,default=None,
                        help="path to directory of chekpoints")
    parser.add_argument("--step",type=int,default=None,
                        help="step to load, if None the final step is loaded")
    parser.add_argument("--n_clusters",type=int,default=None,
                        help= 'n of clusters')
    parser.add_argument("--model",type=str,default="ConvNetQuake",
                        help="model to load")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--output",type=str,default="output/predict",
                        help="dir of predicted events")
    parser.add_argument("--plot", action="store_true",
                     help="pass flag to plot detected events in output")
    args = parser.parse_args()

    main(args)
