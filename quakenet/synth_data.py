import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import quakenet.data_io as data_io
import quakenet.data_conversion as data_conversion


def make_synthetic_data(templates_dir, trace_duration,
                        output_path,
                        max_amplitude,
                        stream_nb = 0,
                        random_scalings=1):

    """ Insert a template and add noise to create a stream. 
    Also create a label stream with 0 if no event and 1 if there
    is an event"""

    template_streams = []
    template_names = []
    for f in os.listdir(templates_dir):
        f = os.path.join(templates_dir, f)
        if os.path.isfile(f) and '.mseed' in f:
            template_streams.append(data_io.load_stream(f))
            template_names.append(f)

    if len(template_streams) < 1:
        raise ValueError('Invalid path "{}", contains no .mseed template_streams'.format(template_streams_dir))

    print '+ Creating synthetic data from {} template_streams'.format(len(template_streams))

    # TODO(mika): check that all the template_streams have the same meta
    sampling_rate = template_streams[0][0].stats.sampling_rate
    n_channels = len(template_streams[0])
    stream_info = {}
    stream_info['station'] = template_streams[0][0].stats.station
    stream_info['network'] = template_streams[0][0].stats.network
    stream_info['channels'] = [tr.stats.channel for tr in template_streams[0]]

    out_npts = int(sampling_rate * trace_duration)

    # Centered, normalized and windowed template signal
    templates = get_normalized_templates(template_streams)
    original_dtype = templates[0].dtype
    # TODO: check range of original data, what should it be?

    print '  Sampling rate {} Hz'.format(sampling_rate)

    # Sample-accurate intervals between events (1min to 1h delay).
    # Slightly oversample so that the we cover the full trace length.
    low = 60 * sampling_rate
    high = 60 * 60 * sampling_rate
    mean = (low + high) / 2.0
    intervals = np.random.randint(low=low,
                                  high=high,
                                  size=int(1.5 * out_npts / mean))

    # Keep only events in the time interval of the synthetic data
    events = np.cumsum(intervals)
    maxpts = max([t.shape[1] for t in templates])  # Max length of a template
    events = events[events < out_npts - maxpts]

    # Random scaling and event
    if random_scalings==1:
        scales = max_amplitude * np.random.uniform(size=len(events))
    else:
        scales = max_amplitude * np.ones(len(events))
    template_ids = np.random.randint(0, len(templates), size=len(events))

    # Noise floor (WGN)
    # TODO(mika): replace with more realistic Earth noise
    # TODO(tibo): extract few  seconds of recording in CA
    noise = np.random.normal(size=(n_channels, out_npts))

    # Create a signal with noise
    signal = np.copy(noise)

    # Create label stream
    label = np.zeros((1, out_npts))

    # Measure SNR
    A_noise = 0
    A_signal = 0

    # Add events to the signal and 1 in label when there is an event
    for e, s, tid in zip(events, scales, template_ids):
        # TODO: adds random shift to model the propagation time
        t = templates[tid]
        npts = t.shape[1]
        signal[:, e:e + npts] += s * t
        label[:, e:e + npts] = 1

        A_noise += np.sum(np.square(noise[:, e:e + npts]))
        A_signal += np.sum(np.square(s * t))

    snr = A_signal/A_noise
    snr = 10*np.log10(snr)

    print "+ Converting back to {}".format(original_dtype)
    signal = signal.astype(original_dtype)
    label = label.astype(original_dtype)
    out_stream = data_conversion.array2stream(signal, sampling_rate,
                                              stream_info)
    out_label = data_conversion.array2stream(label, sampling_rate,
                                              stream_info)

    print '+ Generated {} events in {}s'.format(len(events), trace_duration)
    print '+ SNR on events windows: {:.1f} dB'.format(snr)


    # Prepare catalog
    starttime = out_stream[0].stats.starttime
    events_time = [starttime + e * 1.0 / sampling_rate for e in events]

    meta = {
        'sampling_rate': sampling_rate,
        'n_events': len(events),
        'snr': snr,
        'max_amplitude': max_amplitude,
        'templates': template_names,
    }

    # Save catalog and stream
    catalog_fmt = 'catalog_{:03d}.csv'
    stream_fmt = 'stream_{:03d}.mseed'
    label_fmt = 'label_{:03d}.mseed'
    meta_fmt = 'meta_{:03d}.json'
    catalog_path = os.path.join(output_path, catalog_fmt.format(stream_nb))
    stream_path = os.path.join(output_path, stream_fmt.format(stream_nb))
    meta_path = os.path.join(output_path, meta_fmt.format(stream_nb))
    label_path = os.path.join(output_path, label_fmt.format(stream_nb))

    print '+ Saving to disk'
    data_io.write_catalog(events_time, catalog_path)
    data_io.write_stream(out_stream, stream_path)
    data_io.write_stream(out_label, label_path)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def get_normalized_templates(template_streams):
    templates = []
    for ts in template_streams:
        t = data_conversion.stream2array(ts)
        t = t.astype(np.float32)
        t -= np.mean(t, axis=1, keepdims=True)
        template_max = np.amax(np.abs(t))
        t /= template_max
        t *= np.blackman(ts[0].stats.npts)
        templates.append(t)
    return templates
