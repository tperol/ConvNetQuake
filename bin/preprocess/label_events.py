#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import sys

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import obspy.core.utcdatetime as utc
import numpy as np
import pandas as pd

import quakenet.data_io as data_io

class MainWindow(QtGui.QMainWindow):
  def __init__(self, output):
    super(MainWindow, self).__init__()

    self.mw = MainWidget(output, parent=self)
    self.setCentralWidget(self.mw)
    self.mw.setFocus()
    self.show()

class MainWidget(QtGui.QWidget):
  def __init__(self, output, parent=None):
    super(MainWidget, self).__init__(parent=parent)

    self.load_thread = None
    self.stream = None
    self.catalog = None
    self.filtered_catalog = None
    self.num_events = 0
    self.event_idx = -1
    self.output = output

    self.statusBar = self.parent().statusBar()

    self.layout = QtGui.QVBoxLayout()
    self.setLayout(self.layout)
    self.resize(800,800)

    self.set_buttons()
    self.set_graphics_view()

  def set_buttons(self):
    self.button_layout = QtGui.QHBoxLayout()
    self.prev_button = QtGui.QPushButton('Previous Event')
    self.next_button = QtGui.QPushButton('Next Event')
    # self.catalog_button = QtGui.QPushButton("Save Catalog")
    self.next_button.clicked.connect(self.next_event)
    self.prev_button.clicked.connect(self.prev_event)
    # self.catalog_button.clicked.connect(self.save_catalog)

    self.button_layout.addWidget(self.prev_button)
    self.button_layout.addWidget(self.next_button)
    # self.button_layout.addWidget(self.catalog_button)

    self.layout.addLayout(self.button_layout)

  def set_graphics_view(self):
    self.win = pg.GraphicsLayoutWidget()
    self.layout.addWidget(self.win)

    self.trace_x = self.win.addPlot(title="X")
    self.win.nextRow()
    self.trace_y = self.win.addPlot(title="Y")
    self.trace_y.setXLink(self.trace_x)
    self.trace_y.setYLink(self.trace_x)
    self.win.nextRow()
    self.trace_z = self.win.addPlot(title="Z")
    self.trace_z.setXLink(self.trace_x)
    self.trace_z.setYLink(self.trace_x)

  def keyPressEvent(self, e):
    if e.key() == QtCore.Qt.Key_Right:
      self.next_event()
    if e.key() == QtCore.Qt.Key_Left:
      self.prev_event()

  def save_waveform(self):
    start_time = self.lrx.getRegion()[0]
    end_time = self.lrx.getRegion()[1]
    if start_time < end_time:
      self.start_times[self.event_idx] = start_time
      self.end_times[self.event_idx] = end_time
    else:
      self.start_times[self.event_idx] = end_time
      self.end_times[self.event_idx] = start_time
      
  def save_catalog(self):
    self.filtered_catalog.insert(0,"start_times", self.start_times)
    self.filtered_catalog.insert(0,"end_times", self.end_times)
    self.filtered_catalog.to_csv(self.output)

  def next_event(self):
    self.save_waveform()
    self.event_idx = min(self.event_idx+1, self.num_events)
    if self.event_idx == self.num_events:
      self.save_catalog()
    self.plot_trace()

  def prev_event(self):
    self.event_idx = max(self.event_idx-1, 0)
    self.plot_trace()

  def plot_trace(self):
    event_time = self.filtered_catalog.utc_timestamp.values[self.event_idx]
    self.statusBar.showMessage('Event {} of {}: {}'.format(
      self.event_idx+1, self.num_events, utc.UTCDateTime(event_time)))

    window_sz = 20 # in sec
    utc_time = utc.UTCDateTime(event_time)
    start = utc_time
    end = utc_time+window_sz
    local_stream = self.stream.slice(start, end)
    local_stream.filter('highpass', freq=2.0)

    sample_rate = local_stream[0].stats.sampling_rate
    npts = local_stream[0].stats.npts

    event_sample = (utc_time-start)*sample_rate

    n_traces = len(local_stream)
    n_samples = len(local_stream[0].data)
    data = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i in range(n_traces):
        data[i, :] = local_stream[i].data[...]
        mean = np.mean(data[i, :])
        data[i, :] -= mean

    self.trace_x.clear()
    self.trace_y.clear()
    self.trace_z.clear()

    self.trace_x.plot(data[0, :], pen=(255,120,120,200))
    self.trace_y.plot(data[1, :], pen=(120,255,120,200))
    self.trace_z.plot(data[2, :], pen=(120,120,255,200))

    self.lrx = pg.LinearRegionItem([event_sample,event_sample+sample_rate*1])
    self.lrx.setZValue(-10)
    self.trace_x.addItem(self.lrx)

    # lry = pg.LinearRegionItem([400,700])
    # lry.setZValue(-10)
    # self.trace_y.addItem(lry)
    #
    # lrz = pg.LinearRegionItem([400,700])
    # lrz.setZValue(-10)
    # self.trace_z.addItem(lrz)
    #
    # regions = [lrx, lry, lrz]
    #
    # def updateRange(lr, regions):
    #   for l in regions:
    #     if l != lr:
    #       l.setRegion(lr.getRegion())
    #
    # # for l in regions:
    # lrx.sigRegionChanged.connect(lambda : updateRange(lrx, regions))
    # lry.sigRegionChanged.connect(lambda : updateRange(lry, regions))
    # lrz.sigRegionChanged.connect(lambda : updateRange(lrz, regions))


  def load_catalog(self, catalog_path):
    self.catalog = data_io.load_catalog(catalog_path)
    self.statusBar.showMessage('Loaded catalog {}.'.format(
      os.path.split(catalog_path)[-1]))

  def load_stream(self, stream_path):
    self.statusBar.showMessage('Loading stream {}... please wait.'.format(
      os.path.split(stream_path)[-1]))
    self.load_thread = StreamLoadingThread(stream_path)
    self.load_thread.finished.connect(self.load_stream_finished)
    self.load_thread.start()

  def load_stream_finished(self):
    self.statusBar.showMessage('Loaded stream {}'.format(
      os.path.split(self.load_thread.stream_path)[-1]))
    self.stream = self.load_thread.stream
    self.load_thread = None

    start_date = self.stream[0].stats.starttime
    end_date = self.stream[0].stats.endtime

    self.filtered_catalog = self.catalog[
        ((self.catalog.utc_timestamp >= start_date)
        & (self.catalog.utc_timestamp < end_date))]

    self.num_events = len(self.filtered_catalog.utc_timestamp.values)
    assert self.num_events > 0
    self.event_idx = 0
    self.start_times = np.zeros(self.num_events,dtype=np.float64) * np.nan
    self.end_times = np.zeros(self.num_events,dtype=np.float64) * np.nan

    self.plot_trace()

class StreamLoadingThread(QtCore.QThread):

  def __init__(self, stream_path):
    super(StreamLoadingThread, self).__init__()
    self.stream_path = stream_path

  def run(self):
    self.stream = data_io.load_stream(self.stream_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--stream', default='data/streams/GSOK029_2-2014.mseed')
  parser.add_argument('--catalog', default='data/hackathon/catalog.csv')
  parser.add_argument('--output', default='data/hackathon/picked_catalog.csv')
  args = parser.parse_args()

  app = QtGui.QApplication([])
  # QtGui.QApplication.setGraphicsSystem('opengl')
  pg.setConfigOptions(antialias=True)
  app.setApplicationName('Event labeler')

  mw = MainWindow(args.output)
  mw_w = mw.mw
  mw_w.load_catalog(args.catalog)
  mw_w.load_stream(args.stream)

  if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
