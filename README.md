
ConvNetQuake
=============

Details about ConvNetQuake can be found in:

Perol., T, M. Gharbi and M. Denolle. Convolutional Neural Network for Earthquake detection and location. [preprint arXiv:1702.02073](https://arxiv.org/abs/1609.03499), 2017.

## Installation

* Install dependencies: `pip install -r requirements.txt`
* Add directory to python path: `./setpath.sh`
* Run tests: `./runtests.sh` (THIS NEED TO BE EXTENDED)

## Data

ConvNetQuake is trained on data from Oklahoman (USA). 
The continuous waveform data are publicly available at https://www.iris.edu/hq/
The earthquake catalog from the [state survey](http://www.ou.edu/ogs.html) can be downloaded [here](http://people.seas.harvard.edu/~tperol/data/catalogs/OK_2014-2015-2016.csv).
The earthquake catalog compiled by Benz et al. 2015 is available at [here](http://people.seas.harvard.edu/~tperol/data/catalogs/Benz_catalog.csv).

All the data used in the paper is hosted [here](https://www.dropbox.com/sh/8kx6zo7z1tzze6h/AACIerv_QArpMaOir9Q8vpsBa?dl=0):


We will provide the link to download the generated windows of events and windows of seismic noise soon.
* Download the windows used for training and testing at **insert url** 
* Download the trained models at **insert url**


## 1 - What will you find in this repository ?

ConvNetQuake is a Convolutional Neural Network that detect and locate events from a single waveform.
The dataset consists of continuous waveform data from two stations in Oklahoma: GS.OK029 and GS.OK027. 
The catalogs are from the OGS and Benz et al., 2015.

This repository contains all the codes used to write our paper. For each step, we provide the commands to run.

## 2 - Train ConvNetQuake on a dataset

There are multiple steps in training ConvNetQuake on a dataset of waveforms. 

- Use a catalog of localized events and partition them into clusters. This create a new catalog of events with their labels (cluster index). The script is in `bin/preprocess`.
- Load month long continuous waveform data. Preprocess them (mean removal, normalization). Use the catalog of labeled events to create windows of events from the continuous waveform data. Use an extended catalog to create windows of noise from continuous waveform data. In Perol et al., 2017 we use a catalog of events found using a subspace detection method (Benz et al., 2015). This step is essential to train ConvNetQuake to recognize noise. The codes are in `bin/preprocess`.
- Train ConvNetQuake on the windows created, visualize of the training and evaluate on a test set. The codes are in `bin/`.

### 2.1 - Partition events into clusters

Load the catalog from the state survey of localized events for the year 2014, 2015 and 2016. Filter to keep the events in the region of interest and keep events after 15 February 2014 (the activation date of the two stations used in this study). The script partition the events into clusters using the latitude and longitude. In Perol et al., 2017 we use K-means for clustering. Other methods are also implemented. To do the clustering run:

```shell
./bin/preprocess/cluster_events --src data/catalogs/OK_2014-2015-2016.csv\
--dst data/6_clusters --n_components 6 --model KMeans
```

This clusters the events into 6 clusters using the K-means algorithm. 
We output the catalog of labeled events `catalog_with_cluster_ids.csv` in `data/6_clusters`/ . 
We also create  `clusters_metadata.json` that provide information about the number of events per clusters. 
The code also plot the events in a map. The colored events are the training events, the black events are the events in the test set (July 2014).

The cluster labels range from 0 to N-1 with N the number of clusters. 

### 2.2 Create labeled windows of events

Load a directory of month long stream in .mseed and the catalog of labeled events. The script preprocess the month long streams (remove the mean, normalization). Using the origin time of the event from the catalog and a mean velocity of 5 km/s between the station and the event location, we create a 10 seconds long window of the events.

```shell
./bin/preprocess/create_dataset_events.py --stream_dir data/streams\
--catalog data/6_clusters/catalog_with_cluster_ids.csv \
--output_dir data/6_clusters/events \
--save_mseed True --plot True
```

In this case the tfrecords of the name of the .mseed processed are created in the output directory. Pass `—save_mseed` to create a directory with the windows events created saved in .mseed format. Pass `—plot` to plot the events in .png.

One can do **data augmentation** of the windows of events created. Once tfrecords of labeled event are created, load them with the `data_augmentation.py` script and add Gaussian noise or stretch or shift the signal and generate a new tfrecords from it. To do this, run:

```shell
./bin/preprocess/data_augmentation.py --tfrecords data/agu/detection/train/positive \
--output data/agu/detection/augmented_data/augmented_stetch_std1-2.tfrecords \
--std_factor 1.2
```

This add Gaussian Noise in the data of a standard deviation 1.2 times the one in the window. You can pass various flags. `-plot` plot the created windows. `—compress_data` compress the signal. `—stretch_data` stretch the signal. `—shift_data` shifts the signal (not useful for ConvNetQuake because of the translation invariance of convolutional neural networks). In Perol et al., 2017 we only add Gaussian noise. The other data augmentation techniques do not improve the accuracy of the trained network.

### 2.3 Create windows of noise

Load one month long stream, preprocess them and load an extended catalog to create windows of noise labeled as -1. Note that we will add 1 to all labels when training ConvNetQuake because of the error function choice. 

```shell
./bin/preprocess/create_dataset_noise.py \
--stream_path data/streams/GSOK029_8-2014.mseed \
--catalog data/catalogs/Benz_catalog.csv \
--output_dir data/noise_OK029/noise_august
```

This generates windows of 10 seconds long with a lag of 10 seconds. Check the flags in the code if you want to change this. `-—max_windows` for the maximum number  of windows to generate. For each window, we look at the extended catalog to check if there is an event in it. Note that this extended catalog correspond to the time on the continuous waveform data. No need to account for propagation time. The `—plot` and `—save_mseed` are available.

### 2.4 Train ConvNetQuake and monitor the accuracy on train and test sets

Place the windows of noise in a folder named `negative` and the windows of events in a folder named `positive`. These two folders are placed into the training folder. To train ConvNetQuake run:

```shell
./bin/train --dataset data/6_clusters/train --checkpoint_dir output/convnetquake --n_clusters 6
```

This train the network by feeding an equal amount of positive and negative examples per batch.

In this case the checkpoints will be saved in the checkpoint_dir. In the checkpoints folder, there are checkpoints with saved weights and events for tensorboard. The checkpoints are named after the number of steps done during training. For example `model-500` correspond to the weights after 500 steps of training. The configuration parameters (batch size, display step etc) can be found and changed in `quakenet/config.py`. This script run the model detailed in `quakenet/models.py`. Training is usually done on a GPU.

Note that we provide the trained model hosted at  **insert url**.

During training, there are two things to monitor: the accuracy on the windows of noise and accuracy on windows of events. These can be ran on CPUS since there are not computationally expensive. The scripts to run are:

```shell
./bin/evaluate --checkpoint_dir output/convnetquake/ConvNetQuake \
--dataset data/6_clusters/test_events \
--eval_interval 10 --n_clusters 6 \
--events
```

This evaluate the accuracy on the events of the test set. The program sleeps for 10 second after one evaluation.  The code sleeps until the first checkpoint of convnetquake is saved.

```shell
./bin/evaluate --checkpoint_dir output/convnetquake/ConvNetQuake \
--dataset data/6_clusters/test_noise --eval_interval 10 \
--n_clusters 6 --noise
```

This evaluate the accuracy on the windows of noise.

For the evaluation procedure, there is no need to create a postive and negative folder.

You can visualize the accuracy on the train and test set while the network is training. The accuracy for detection and for localization is implemented. Run

```shell
tensorboard --logdir output/convnetquake/ConvNetQuake
```



![Monitoring detection accuracy on train and test sets during training of ConvNetQuake](./figures/training.png)

You can also visualize the network architecture

![ConvNetQuake architecture](./figures/architecture.png)

## 3 - Detecting and localizing events in continuous waveform data

There are two methods for detecting events from continuous waveform data. 
The first one is relatively slow, it loads a .mseed and generate windows. While the windows are generated they are fed to ConvNetQuake that makes the classification. A faster method does the classification from tfrecords. First windows, a generated and saved into tfrecords. Then the tfrecords are analyzed and classified by ConvNetQuake to create a catalog of events.
This second methods analyze one month of data in 4 min on a MacbookPro.

### 3.1 From .mseed

Run:
```shell
TODO
```

### 3.2 From tfrecords (faster)
First, the windows are generated from a .mseed and stored into a tfrecords.

```shell
./bin/preprocess/convert_stream_to_tfrecords.py \
--stream_path data/streams/GSOK029_7-2014.mseed \
--output_dir  data/tfrecord \
--window_size 10 --window_step 11 \
--max_windows 5000
```
See the code for a documentation on the flags to pass.
This code can be parallelized easily to speed up the process.

Then, the detection from the windows stored in tfrecords are made with

```shell
TODO
```

## 4 - Visualization of data

The codes for vizualization can be found in `bin/viz`.

### 4.1 - Print the cluster Id and number of events

```shell
./bin/viz/print_clusterid_from_tfrecords.py \
--data_path data/agu/detection/train/positive \
--windows 40000
```

If the number of events in the tfrecords in the `data_path` directory is lower than 40000, the number of events is printed. The cluster ids are also printed


### 4.2 - Visualize windows from tfrecords

```shell
./bin/viz/plot_windows_from_tfrecords.py \
--data_path data/tfrecords/GSOK029_2-2014 \
--output_path output/viz --windows 100
```

Load tfrecords from a directory and plot the windows.

### 4.3 - Plot events from a .mseed stream

```shell
./bin/viz/plot_events_in_stream.py \
--catalog data/6_clusters/catalog_with_cluster_ids.csv  \
--stream data/streams/GSOK029_7-2014.mseed \
--output check_this_out/events_in_stream \
--with_preprocessing
```

Load a .mseed with a catalog and plot the windows of events.

### 4.4 - Visualize mislabeled windows

To visualized the mislabeled windows from a net on a probabilistic map (see Figure ? of Perol et al., 2017)
```shell
./bin/viz/misclassified_loc.py \
--dataset data/mseed_events \
--checkpoint_dir model/convnetquake \
--output wrong_windows --n_clusters 6
```

## 5 - Generate a synthetic stream/catalog pair

- To generate 3600s of synthetic signal do:

```shell
./bin/create_synthetics --templates_dir data/streams/templates/\
--output_path data/synth --trace_duration 3600
```

This will output a .mseed stream and a .csv catalog to `data/synth.`
The signal is a a white gaussian noise. Events are generated randomly with 
a uniform distance between events sampled from [1min, 1h]. These are inserted
as a scaled copy of the template event

`templates_dir` should contain .mseed files with individual source templates e.g.
`templates_dir/templates01.mseed`, `templates_dir/templates02.mseed` , etc.

You can use this data as any other dataset/catalog pair to generate .tfrecords.


6 - Template matching method
------------------------
To train the template matching method (find the best beta parameter) on a
training set (one stream, one template, one catalog) and perform a test on a
test set ang get the score:

```shell
./bin/template_matching --train_data_path data/synth/stream.mseed \
--train_template_path data/streams/template_0.mseed \
--train_catalog_path data/synth/catalog.csv \
--test_data_path data/synth/stream.mseed \
--test_template_path data/streams/template_0.mseed \
--test_catalog_path data/synth/catalog.csv
```
It is possible to avoid training and only test on a stream. In this case beta
= 8.5. The command is:

```shell
./bin/template_matching --test_data_path data/synth/stream.mseed \
--test_template_path data/streams/template_0.mseed \
--test_catalog_path data/synth/catalog.csv
```


7 - Codebase
--------

Stream and catalog loaders are in `quakenet/data_io.py`.

`quakenet/data_pipeline.py` contains all the
preprocessing/conversion/writing/loading of the data used by the
network.

`quakenet/synth_data.py` contains the code that generates synthetic data from
a set of templates 

`quakenet/models.py` contains the tensorflow code for the architecture of ConvNetQuake

Tensorflow base model and layers can be found in `tflib` repository.
