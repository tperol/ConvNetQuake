import unittest
from quakenet import data_io
from quakenet.test import settings
from os import path

def test_load_stream():
    tpath = path.join(settings.TEST_DATA_DIR,
                      'event_stream.mseed')
    stream = data_io.load_stream(tpath)
    assert(len(stream) == 3)
    assert(stream[0].stats.sampling_rate == 100 )
    assert(stream[0].stats.npts == 1001 )

def test_write_stream()

def test_load_catalog():
    pass
    # TODO(mika): create specific test objects
    # cpath = path.join(settings.DATA_DIR,
    #                      'catalogs',
    #                      'Guthrie_catalog.csv')
    # assert(path.exists(cpath))
    # data_io.load_catalog(cpath)
