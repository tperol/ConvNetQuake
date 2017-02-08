#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : plot_windows_from_tfrecords.py
# Creation Date : 04-12-2016
# Last Modified : Sun Dec  4 13:42:04 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""
plot traces and cluster id
and save png files of the windows from a directory of tfrecords
e.g.,
./bin/viz/windows --data_path data/tfrecords
--output_path data/eval_tfrecords/viz --windows 12
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm

from quakenet import data_pipeline as dpp
import quakenet.config as config


flags = tf.flags
flags.DEFINE_string('data_path',
                    None, 'path to the records containing the windows.')
flags.DEFINE_string('output_path',
                    None, 'path to save the checkpoints and summaries.')
flags.DEFINE_integer('windows', 100, 'number of windows to display.')
FLAGS = flags.FLAGS


def main(_):

    cfg = config.Config()
    cfg.batch_size = 1

    # Make dirs
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    data_pipeline = dpp.DataPipeline(FLAGS.data_path,
                                     config=cfg,
                                     is_training=False)
    samples = data_pipeline.samples
    labels = data_pipeline.labels

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print '+ Plotting {} windows'.format(FLAGS.windows)
        for i in tqdm(range(FLAGS.windows)):
            sample, label = sess.run([samples, labels])
            sample = np.squeeze(sample, axis=(0,))
            target = np.squeeze(label, axis=(0,))

            plt.clf()
            fig, ax = plt.subplots(3, 1)
            for t in range(sample.shape[1]):
                ax[t].plot(sample[:, t])
                ax[t].set_xlabel('time (samples)')
                ax[t].set_ylabel('amplitude')
            ax[0].set_title('window {:04d}, cluster_id: {}'.format(i, target))
            plt.savefig(os.path.join(FLAGS.output_path,
                                     'window_{:04d}.pdf'.format(i)))
            plt.close()

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
