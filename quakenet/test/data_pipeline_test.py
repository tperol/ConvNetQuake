import unittest
from quakenet import data_pipeline as dpp
from quakenet.test import settings
import os
from os import path
from obspy.core.utcdatetime import UTCDateTime

def test_data_generator_is_instantiated():
    wsz = 300
    wstp = 300
    dg = dpp.DatasetGenerator(wsz, wstp)
    assert(dg.win_size == wsz)
    assert(dg.win_step == wstp)


def test_data_generator_generates_samples(tmpdir):
    wsz = 3
    wstp = 3
    dg = dpp.DatasetGenerator(wsz, wstp)

    # TODO(mika): create specific test objects
    tpath = path.join(settings.DATA_DIR,
                      'streams',
                      'GSOK029_2-2014.mseed')
    # TODO(mika): create specific test objects
    cpath = path.join(settings.DATA_DIR,
                         'catalogs',
                         'Guthrie_catalog.csv')
    
    output_path = str(tmpdir)
    # TODO(mika): actually test it...
    # dg.generate(tpath,
    #             cpath,
    #             output_path)


