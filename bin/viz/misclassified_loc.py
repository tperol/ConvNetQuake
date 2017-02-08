#!/usr/bin/env python
# -------------------------------------------------------------------
# File: misclassified_loc
# Author: Thibaut Perol <tperol@g.harvard.edu>
# Created: 2016-11-16
# ------------------------------------------------------------------#

""" plot misclassified windows with their predicted
and true labels on a risk map
e.g.,
./bin/viz/misclassified_loc --dataset data/6_clusters/see/mseed
--checkpoint_dir output/odyssey/6_clusters/BestOne_100det_71loc --model
ClusterPredictionBest --output here/wrong --n_clusters 6

Note: In this case we have store the test event traces in mseed format in
data/6_clusters/see/mseed using
./bin/data/create_dataset_enents.py \
--stream_dir data/eval_streams/ --catalog
data/6_clusters/catalog_with_cluster_ids.csv  --output_dir data/6_clusters/see
--save_mseed True

misclassified_loc read the mseed file with the name that indicates the label,
the lat and long in order to plot the test event on the risk map produced by
ConvNetQuake
"""

import os
import setproctitle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil

import quakenet.models as models
from quakenet.data_pipeline import DataPipeline
import quakenet.config as config
from quakenet.data_io import load_stream

flags = tf.flags
flags.DEFINE_string('dataset',
                    None, 'path to the records to validate on.')
flags.DEFINE_string('checkpoint_dir',
                    None, 'path to the directory of checkpoints')
flags.DEFINE_integer('step', None, 'step to load')
flags.DEFINE_integer('n_clusters', None, 'n of clusters')
flags.DEFINE_string('model',
                    None, 'model name to load')
flags.DEFINE_string('output', 'output/misclassified',
                    'dir of plotted misclassified windows')
args = flags.FLAGS

# TODO: Allow for variable length of window
print "ATTENTION: FOR NOW WINDOWS' LENGTH = 10 SECONDS"


def fetch_window_and_label(stream_name):
    """Load window stream, extract data and label"""
    stream = load_stream(stream_name)
    data = np.empty((1001, 3))
    for i in range(3):
        data[:, i] = stream[i].data.astype(np.float32)
    data = np.expand_dims(data, 0)
    stream_name = os.path.split(stream_name)[-1]
    label = np.empty((1,))
    label[0] = stream_name.split("_")[1]
    return data, label


def fetch_lat_and_lon(stream):
    lat = stream.split("_")[4]
    lon = stream.split(".mseed")[0].split("_")[-1]
    return float(lat), float(lon)


def fetch_streams_list(datadir):
    """Get the list of streams to analyze"""
    fnames = []
    for root, dirs, files in os.walk(datadir):
        for f in files:
            if f.endswith(".mseed"):
                fnames.append(os.path.join(root, f))
    return fnames


def plot_proba_map(i, lat,lon, clusters, class_prob, label,
                   lat_event, lon_event):

    plt.clf()
    class_prob = class_prob / np.sum(class_prob)
    assert np.isclose(np.sum(class_prob),1)
    risk_map = np.zeros_like(clusters,dtype=np.float64)
    for cluster_id in range(len(class_prob)):
        x,y = np.where(clusters == cluster_id)
        risk_map[x,y] = class_prob[cluster_id]

    plt.contourf(lon,lat,risk_map,cmap='YlOrRd',alpha=0.9,
                 origin='lower',vmin=0.0,vmax=1.0)
    plt.colorbar()

    plt.plot(lon_event, lat_event, marker='+',c='k',lw='5')
    plt.contour(lon,lat,clusters,colors='k',hold='on')
    plt.xlim((min(lon),max(lon)))
    plt.ylim((min(lat),max(lat)))
    png_name = os.path.join(args.output,
                    '{}_pred_{}_label_{}.eps'.format(i,np.argmax(class_prob),
                                                     label))
    plt.savefig(png_name)
    plt.close()

def main(_):
    setproctitle.setproctitle('quakenet_viz')

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

    cfg = config.Config()
    cfg.batch_size = 1
    cfg.n_clusters = args.n_clusters
    cfg.add = 1
    cfg.n_clusters += 1

    # Remove previous output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    windows_list = fetch_streams_list(args.dataset)

    # stream data with a placeholder
    samples = {
            'data': tf.placeholder(tf.float32,
                                   shape=(cfg.batch_size, 1001, 3),
                                   name='input_data'),
            'cluster_id': tf.placeholder(tf.int64,
                                         shape=(cfg.batch_size,),
                                         name='input_label')
        }

    # set up model and validation metrics
    model = models.get(args.model, samples, cfg,
                       args.checkpoint_dir,
                       is_training=False)
    metrics = model.validation_metrics()

    with tf.Session() as sess:

        model.load(sess, args.step)
        print 'Evaluating at step {}'.format(sess.run(model.global_step))

        step = tf.train.global_step(sess, model.global_step)
        mean_metrics = {}
        for key in metrics:
            mean_metrics[key] = 0

        for n in range(len(windows_list)):

            # Get One stream and label from the list
            stream, cluster_id = fetch_window_and_label(windows_list[n])

            # Get coordinates of the event
            lat_event, lon_event = fetch_lat_and_lon(windows_list[n])

            # Fetch class_proba and label
            to_fetch = [samples['data'],
                        metrics,
                        model.layers['class_prob']]
            feed_dict = {samples['data']: stream,
                         samples['cluster_id']: cluster_id}
            sample, metrics_, class_prob_= sess.run(to_fetch,
                                                    feed_dict)
            # Keep only clusters proba, remove noise proba
            clusters_prob = class_prob_[0,1::]

            # Print Misclassified window
            if metrics_['localization_accuracy'] >= 1.0:
                map_file ='cluster_ids_{}_comp.npy'.format(args.n_clusters)
                clusters_map = np.load(map_file)
                lat = np.load("cluster_ids_{}_comp_lat.npy".format(args.n_clusters))
                lon = np.load("cluster_ids_{}_comp_lon.npy".format(args.n_clusters))
                plot_proba_map(n, lat, lon, clusters_map, clusters_prob,
                               cluster_id, lat_event, lon_event)

            for key in metrics:
                mean_metrics[key] += cfg.batch_size * metrics_[key]

            mess = model.validation_metrics_message(metrics_)
            print '{:03d} | '.format(n) + mess

        for key in metrics:
            mean_metrics[key] /= len(windows_list)

        mess = model.validation_metrics_message(mean_metrics)
        print 'Average | ' + mess

if __name__ == '__main__':
    tf.app.run()
