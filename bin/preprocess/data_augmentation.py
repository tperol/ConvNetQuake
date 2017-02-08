#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : data_augmentation.py
# Creation Date : 03-12-2016
# Last Modified : Sun Dec  4 19:25:18 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------

"""Read some tfrecords (1 epoch),
    proceed to data augmentation and save new tfrecords"""


import os
import numpy as np
from quakenet.data_pipeline import DataPipeline,DataWriter
import tensorflow as tf
import fnmatch
import json
import matplotlib.pylab as plt
import librosa

import quakenet.config as config
import quakenet.data_conversion as data_conversion

flags = tf.flags
flags.DEFINE_string('tfrecords', None,
                    'path to the directory of tfrecords to preprocess.')
flags.DEFINE_string('output', None,
                    'path to the tfrecords file to create')
flags.DEFINE_float("std_factor",1.0,"std of the introduced noise")
flags.DEFINE_boolean("plot",False,
                     "True to plot detected events in output")
flags.DEFINE_boolean("compress_data",False,
                     "True to compress the time series")
flags.DEFINE_boolean("stretch_data",False,
                     "True to stretch the time series")
flags.DEFINE_boolean("shift_data",False,
                     "True to shift the time series")
FLAGS = flags.FLAGS

def add_noise_to_signal(data):
    """Add white noise to the signal"""
    s_mean = np.mean(data,axis=0)
    s_std = np.std(data, axis=0)
    for i in range(data.shape[1]):
        data[:,i] += np.random.normal(loc=s_mean[i],
                                      scale=FLAGS.std_factor * s_std[i],
                                      size=data.shape[0])
    return data

def compress_signal(data):
    compressed_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        y = librosa.effects.time_stretch(data[:,i],0.4)
        compressed_data[:,i] = y[0:2001:2]
    assert compressed_data.shape == (1001,3)
    return compressed_data

def stretch_signal(sample):
    #TODO
    pass

def shift_signal(data):
    shifted_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        shifted_data[:,i] = librosa.effects.pitch_shift(data[:,i],100,3)
    assert shifted_data.shape == (1001,3)
    return shifted_data


def convert_np_to_stream(data):
    data = data.transpose()
    sampling_rate = 100
    n_channels = data.shape[1]
    stream_info = {}
    stream_info['station'] = 'OK029'
    stream_info['network'] = 'GS'
    stream_info['channels'] = ['HH1','HH2','HHZ']
    out_stream = data_conversion.array2stream(data, sampling_rate,
                                              stream_info)
    return out_stream

def plot_true_and_augmented_data(sample,noised_sample,label,n_examples):
    output_dir = os.path.split(FLAGS.output)[0]
    # Save augmented data
    plt.clf()
    fig, ax = plt.subplots(3,1)
    for t in range(noised_sample.shape[1]):
        ax[t].plot(noised_sample[:,t])
        ax[t].set_xlabel('time (samples)')
        ax[t].set_ylabel('amplitude')
    ax[0].set_title('window {:03d}, cluster_id: {}'.format(n_examples,label))
    plt.savefig(os.path.join(output_dir, "augmented_data",
                            'augmented_{:03d}.pdf'.format(n_examples)))
    plt.close()

    # Save true data
    plt.clf()
    fig, ax = plt.subplots(3,1)
    for t in range(sample.shape[1]):
        ax[t].plot(sample[:,t])
        ax[t].set_xlabel('time (samples)')
        ax[t].set_ylabel('amplitude')
    ax[0].set_title('window {:03d}, cluster_id: {}'.format(n_examples,label))
    plt.savefig(os.path.join(output_dir, "true_data",
                            'true__{:03d}.pdf'.format(n_examples)))
    plt.close()


def main(_):

    if FLAGS.stretch_data:
        print "ADD NOISE AND STRETCH DATA"
    if FLAGS.compress_data:
        print "ADD NOISE AND COMPRESS DATA"
    if FLAGS.shift_data:
        print "ADD NOISE AND SHIFT DATA"

    # Make dirs
    output_dir= os.path.split(FLAGS.output)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if FLAGS.plot:
        if not os.path.exists(os.path.join(output_dir,"true_data")):
            os.makedirs(os.path.join(output_dir,"true_data"))
        if not os.path.exists(os.path.join(output_dir,"augmented_data")):
            os.makedirs(os.path.join(output_dir,"augmented_data"))


    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_epochs = 1


    data_pipeline = DataPipeline(FLAGS.tfrecords,
                                     config=cfg,
                                     is_training=False)
    samples = data_pipeline.samples
    labels = data_pipeline.labels

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners(coord=coord)

        output_tfrecords = FLAGS.output
        writer = DataWriter(output_tfrecords)
        n_examples = 0
        while True:
            try:
                sample, label = sess.run([samples,labels])
                sample = np.squeeze(sample,axis=0)
                label = label[0]

                noised_sample = add_noise_to_signal(np.copy(sample))
                if FLAGS.compress_data:
                    noised_sample = compress_signal(noised_sample)
                if FLAGS.stretch_data:
                    noised_sample = stretch_signal(noised_sample)
                if FLAGS.shift_data:
                    noised_sample = shift_signal(noised_sample)

                if FLAGS.plot:
                    plot_true_and_augmented_data(sample,noised_sample,
                                                label,n_examples)


                stream = convert_np_to_stream(noised_sample)
                writer.write(stream,label)

                n_examples +=1

            except KeyboardInterrupt:
                print 'stopping data augmentation'
                break

            except tf.errors.OutOfRangeError:
                print 'Augmentation completed ({} epochs, {} examples seen).'\
                                .format(cfg.n_epochs,n_examples-1)
                break


        writer.close()
        coord.request_stop()
        coord.join(threads)



if __name__=="__main__":
    tf.app.run()
