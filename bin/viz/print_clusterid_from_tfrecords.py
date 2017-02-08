#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : print_clusterid_from_tfrecords.py
# Creation Date : 04-12-2016
# Last Modified : Sun Jan  8 23:28:58 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""
Load all tfrecords in a directory, print their cluster_id, spit the number
of windows seen
e.g.,
./bin/viz/print_cluster_id --data_path data/50_clusters/eval_tfrecords
--windows 3000
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import quakenet.config as config
from obspy.core.utcdatetime import UTCDateTime

from quakenet import data_pipeline as dpp

flags = tf.flags
flags.DEFINE_string('data_path',
        None, 'path to the records containing the windows.')
flags.DEFINE_integer('windows', 10, 'number of windows to display.')
FLAGS = flags.FLAGS


def main(_):


    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_epochs = 1


    data_pipeline = dpp.DataPipeline(FLAGS.data_path,
                                     config=cfg,
                                     is_training=False)
    samples = data_pipeline.samples
    labels = data_pipeline.labels
    start_time = data_pipeline.start_time
    end_time = data_pipeline.end_time

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in (range(FLAGS.windows)):
                to_fetch= [samples, labels, start_time, end_time]
                sample, label, starttime, endtime = sess.run(to_fetch)
                # assert starttime < endtime
                print('starttime {}, endtime {}'.format(UTCDateTime(starttime),
                                                        UTCDateTime(endtime)))
                print("label", label[0])
                sample = np.squeeze(sample, axis=(0,))
                target = np.squeeze(label, axis=(0,))
        except tf.errors.OutOfRangeError:
            print 'Evaluation completed ({} epochs).'.format(cfg.n_epochs)

        print "{} windows seen".format(i+1)
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
