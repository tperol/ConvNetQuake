import os.path

join = os.path.join
dirname = os.path.dirname
split = os.path.split
abspath = os.path.abspath

BASE_DIR = split(split(dirname(abspath(__file__)))[0])[0]
TEST_DATA_DIR = join(BASE_DIR, 'testdata')
