#!/usr/bin/env python
# encoding: utf-8
# Created: 2016-10-25
# -------------------------------------------------------------------
# File:  evaluate
# Author: Thibaut Perol <tperol@g.harvard.edu>
# Created: 2016-11-16
# ------------------------------------------------------------------#

""" Test a model on tfrecords"""

import argparse
import os
import time
import json
import pandas as pd

import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from obspy.core import UTCDateTime

import quakenet.models as models
from quakenet.data_pipeline import DataPipeline
import quakenet.config as config


def main(args):
    
    if args.n_clusters == None:
        raise ValueError('Define the number of clusters with --n_clusters')
    if not args.noise and not args.events:
        raise ValueError("Define if evaluating accuracy on noise or events")

    # Directory in which the evaluation summaries are written
    if args.noise:
        summary_dir =  os.path.join(args.checkpoint_dir,"noise")
    if args.events:
        summary_dir =  os.path.join(args.checkpoint_dir,"events")
    if args.save_false:
        false_start = []
        false_end = []
        false_origintime =[]
        false_dir = os.path.join("output","false_predictions")
        if not os.path.exists(false_dir):
            os.makedirs(false_dir)

    while True:
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if args.eval_interval < 0 or ckpt:
            print 'Evaluating model'
            break
        print  'Waiting for training job to save a checkpoint'
        time.sleep(args.eval_interval)

    cfg = config.Config()
    if args.noise:
        cfg.batch_size = 256
    if args.events:
        cfg.batch_size = 1
    if args.save_false:
        cfg.batch_size = 1
    cfg.n_epochs = 1
    cfg.add = 1
    cfg.n_clusters= args.n_clusters
    cfg.n_clusters +=1

    while True:
        try:
            # data pipeline
            data_pipeline = DataPipeline(args.dataset, config=cfg, 
                                            is_training=False)
            samples = {
                'data': data_pipeline.samples,
                'cluster_id': data_pipeline.labels,
                "start_time": data_pipeline.start_time,
                "end_time": data_pipeline.end_time}


            # set up model and validation metrics
            model = models.get(args.model, samples, cfg,
                               args.checkpoint_dir, 
                               is_training=False)
            metrics = model.validation_metrics()
            # Validation summary writer
            summary_writer = tf.train.SummaryWriter(summary_dir, None)

            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                tf.initialize_local_variables().run()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                model.load(sess,args.step)
                print  'Evaluating at step {}'.format(sess.run(model.global_step))

                step = tf.train.global_step(sess, model.global_step)
                mean_metrics = {}
                for key in metrics:
                    mean_metrics[key] = 0

                n = 0
                pred_labels = np.empty(1)
                true_labels = np.empty(1)
                while True:
                    try:
                        to_fetch  = [metrics,
                                     model.layers["class_prediction"],
                                     samples["cluster_id"],
                                     samples["start_time"],
                                     samples["end_time"]]
                        metrics_, batch_pred_label, batch_true_label, starttime, endtime = sess.run(to_fetch)

                        batch_pred_label -=1 
                        pred_labels = np.append(pred_labels,batch_pred_label)
                        true_labels = np.append(true_labels,
                                                 batch_true_label)

                        # Save times of false preds
                        if args.save_false and \
                                batch_pred_label != batch_true_label:
                            print "---False prediction---"
                            print UTCDateTime(starttime), UTCDateTime(endtime)
                            false_origintime.append((starttime[0]+endtime[0])/2)
                            false_end.append(UTCDateTime(endtime))
                            false_start.append(UTCDateTime(starttime))

                        # print  true_labels
                        for key in metrics:
                            mean_metrics[key] += cfg.batch_size*metrics_[key]
                        n += cfg.batch_size

                        mess = model.validation_metrics_message(metrics_)
                        print '{:03d} | '.format(n)+mess

                    except KeyboardInterrupt:
                        print 'stopping evaluation'
                        break

                    except tf.errors.OutOfRangeError:
                        print 'Evaluation completed ({} epochs).'.format(cfg.n_epochs)
                        print "{} windows seen".format(n)
                        break

                if n > 0:
                  for key in metrics:
                    mean_metrics[key] /= n
                    summary = tf.Summary(value=[tf.Summary.Value(
                      tag='{}/val'.format(key), simple_value=mean_metrics[key])])
                    if args.save_summary:
                        summary_writer.add_summary(summary, global_step=step)

                summary_writer.flush()

                mess = model.validation_metrics_message(mean_metrics)
                print 'Average | '+mess

                if args.eval_interval < 0:
                  print 'End of evaluation'
                  break

            tf.reset_default_graph()
            print 'Sleeping for {}s'.format(args.eval_interval)
            time.sleep(args.eval_interval)

        finally:
            print 'joining data threads'
            coord.request_stop()


    if args.save_false:
        false_preds = {}
        false_preds["start_time"] = false_start
        false_preds["end_time"] = false_end
        false_preds["origintime"] = false_origintime
        # false_preds = np.array((false_start, false_end)).transpose()[0]
        # print 'shape', false_preds.shape
        df =  pd.DataFrame(false_preds) 
        df.to_csv(os.path.join(false_dir,"false_preds.csv"))
    pred_labels = pred_labels[1::]
    true_labels = true_labels[1::]
    # np.save("output/pred_labels_noise.npy",pred_labels)
    # np.save("output/true_labels_noise.npy",true_labels)
    print "---Confusion Matrix----"
    print confusion_matrix(true_labels,pred_labels)

    coord.join(threads)

    # if args.save_false:
    #     false_preds = np.array((false_start, false_end)).transpose()
    #     df =  pd.Dataframe(false_preds, columns=["start_time, end_time"])
    #     df.to_csv(os.path.join(false_dir,"false_preds.csv")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default=None,
                        help='path to the recrords to evaluate')
    parser.add_argument('--checkpoint_dir',default='model/convnetquake',
                        type=str, help='path to checkpoints directory')
    parser.add_argument('--step',type=int,default=None,
                        help='step to load')
    parser.add_argument('--model',type=str,default='ConvNetQuake',
                        help='model to load')
    parser.add_argument('--eval_interval',type=int,default=-1,
                        help='sleep time between evaluations')
    parser.add_argument('--n_clusters',type=int,default=None,
                        help='numbe rof clusters in dataset')
    parser.add_argument('--save_summary',type=bool,default=True,
                        help='True to save summary in tensorboard')
    parser.add_argument('--noise', action='store_true',
                        help='pass this flag if evaluate acc on noise')
    parser.add_argument('--events', action='store_true',
                        help='pass this flag if evaluate acc on events')
    parser.add_argument("--save_false", action="store_true",
                        help="pass this flag to save times of false preds")
    parser.set_defaults(profiling=False)
    
    args = parser.parse_args()
    main(args)
