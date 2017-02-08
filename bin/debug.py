#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : debug.py
# Creation Date : 05-12-2016
# Last Modified : Sat Jan 21 12:49:38 2017
# Author: Michael Gharbi <gharbi@mit.edu>
# -------------------------------------------------------------------

import argparse
import os
import setproctitle
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

import quakenet.models as models
import quakenet.data_pipeline as dp
import quakenet.config as config

def main(args):
  setproctitle.setproctitle('quakenet_debug')

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  if args.n_clusters == None:
    raise ValueError('Define the number of clusters with --n_clusters')

  cfg = config.Config()
  cfg.batch_size = 1
  cfg.n_epochs = 1
  cfg.add = 2
  cfg.n_clusters = args.n_clusters
  cfg.n_clusters +=1

  # data pipeline
  data_pipeline = dp.DataPipeline(args.dataset, cfg, False)

  samples = {
    'data': data_pipeline.samples,
    'cluster_id': data_pipeline.labels
    }

  # model
  model_name = args.model
  model = models.get(model_name, samples,
                     cfg, args.checkpoint_dir, is_training=False)

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    tf.initialize_local_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    model.load(sess)
    step = sess.run(model.global_step)
    print  'Debugging at step {}'.format(step)
    # summary_writer = tf.train.SummaryWriter(model.checkpoint_dir, None)

    activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
    weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
    biases = tf.get_collection(tf.GraphKeys.BIASES)

    toget = {}
    toget['0_input'] = model.inputs['data']
    for i, a in enumerate(activations):
      name = a.name.replace('/', '_').replace(':', '_')
      toget['{}_{}'.format(i+1, name)] = a

    for it in range(10):
      print 'running session'
      fetched = sess.run(toget)
      print fetched

      print it
      for f in fetched:
        d = fetched[f]
        d = np.squeeze(d, axis=0)

        plt.figure()
        if len(d.shape) == 2:
          for i in range(d.shape[1]):
            plt.plot(d[:, i])
          # tot_mean = np.mean(np.mean(d,axis=1),axis=0)
          # plt.plot(np.mean(d,axis=1) / tot_mean)
        plt.savefig(os.path.join(args.output, '{}_{}.pdf'.format(it, f)))
        plt.clf()

      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_dir', type=str,
          default='model/convnetquake')
  parser.add_argument('--dataset', type=str, default='data/dummy')
  parser.add_argument('--model', type=str, default='ConvNetQuake')
  parser.add_argument('--output', type=str, default='output/debug')
  parser.add_argument('--n_clusters',type=int,default='None')
  args = parser.parse_args()

  main(args)
