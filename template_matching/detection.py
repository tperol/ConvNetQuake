import numpy as np
from quakenet.data_io import load_stream, load_catalog
import os
from eqcorrscan.core import match_filter as mf


class DetectionTemplateMatching(object):
    """ Detect events using the template matching method implemented in
    EQcorrscan:
    https://github.com/calum-chamberlain/EQcorrscan

    Parameters:
        dataset_path: mseed. path to a stream to analyze
        template_path: mseed. path to a template to look for in the stream
    Attributes:
        nb_day: int64. number of day in the training  stream
    """

    def __init__(self, beta_grid=[7.5, 8.0, 8.5, 9.0]):
        self._betas = beta_grid

    def _split_stream_into_day_streams(self, stream):

        start_time = stream[0].stats.starttime.timestamp
        end_time = stream[0].stats.endtime.timestamp
        delta = stream[0].stats.delta
        self.nb_day = int((end_time - start_time) / (3600 * 24.0))
        day_streams = []
        if self.nb_day > 0:
            for day in range(self.nb_day):
                day_stream = stream.slice(start_time, start_time +
                                          3600 * 24.0 - delta).copy()
                day_streams.append(day_stream)
                start_time += 3600 * 24.0
        else:
            day_streams.append(stream)
        return day_streams

    def fit(self, dataset_path, template_path,
            catalog_path):
        """ Detect events in stream for various beta and find the optimal
        beta parameter
        """
        template = load_stream(template_path)
        stream = load_stream(dataset_path)
        templates = [template]
        template_name = [os.path.split(template_path)[-1].split('.')[0]]

        day_streams = self._split_stream_into_day_streams(stream)
        detection_results = np.zeros(len(self._betas))
        for k in range(len(self._betas)):
            beta = self._betas[k]
            print '------'
            print ' + Running template matching method for beta =',beta
            detections = []
            for st in day_streams:
                detections.append(mf.match_filter(template_names=template_name,
                                                  template_list=templates,
                                                  st=st, threshold=beta,
                                                  threshold_type='MAD',
                                                  trig_int=1.0,
                                                  plotvar=False))
            # Flatten list of detections
            detections = [d for detection in detections for d in detection]
            false_pos, false_neg = self.score(detections, catalog_path)
            print 'FP: {}, FN: {}'.format(false_pos, false_neg)
            detection_results[k] = false_pos + false_neg
        self.beta = self._betas[np.argmin(detection_results)]

    def score(self, detections, catalog_path, lag_allowed=1.0):
        """ Calculate the number of False and Missed detections
        Parameters:
            detections: list. List of timestamps of detected events
            catalog_path: csv file. Path to the catalog of events and their
                          timestamps
            lag_allowed: float. time lag between a cataloged and detected
            event to be considered as a true detection
        Returns:
            false_pos: int. Number of false detections
            false_neg: int. Number of missed detections
        """
        catalog = load_catalog(catalog_path)
        events = catalog.utc_timestamp

        detection_times = [detection.detect_time.timestamp
                           for detection in detections]
        detection_results = [False] * len(detection_times)
        for d in xrange(len(detection_times)):
            detected_event = detection_times[d]
            for event_time in events:
                if np.abs(detected_event - event_time) <= lag_allowed:
                    detection_results[d] = True
        if len(detection_times)>0:
            false_pos = (~np.array(detection_results)).sum()
            false_neg = len(events) - sum(detection_results)
            return false_pos, false_neg
        else:
            return 0, len(events)

    def predict(self, dataset_path, template_path,
                catalog_path):
        template = load_stream(template_path)
        stream = load_stream(dataset_path)
        try:
            print self.beta
        except AttributeError:
            self.beta = 8.5
        templates = [template]
        template_name = [os.path.split(template_path)[-1].split('.')[0]]

        day_streams = self._split_stream_into_day_streams(stream)
        print '-------'
        print ' + Running template matching method on test set'
        detections = []
        for st in day_streams:
            detections.append(mf.match_filter(template_names=template_name,
                                              template_list=templates,
                                              st=st, threshold=self.beta,
                                              threshold_type='MAD',
                                              trig_int=1.0,
                                              plotvar=False))

        # Flatten list of detections
        self.detections = [d for detection in detections for d in detection]
