__author__ = "Vitaly Butoma"
import theano; print(theano.config.openmp)
import os
from noise_reduction import NoiseReduction
from copy_files import _get_file_by_key, _upload_file_by_path
import time
v = NoiseReduction()

TEMP_FILE = 'temp.wav'
OUTPUT_FILE = 'res.wav'

if __name__ == "__main__":
    bucket = os.environ.get('BUCKET')
    filename = os.environ.get('FILENAME')
    # bucket = 'noise-reduction-data'
    # filename = '1.wav'
    print(bucket, filename)
    start_time = time.time()
    _get_file_by_key(filename, TEMP_FILE)
    print('loaded', time.time() - start_time)
    v.solve_big_file(file_path=TEMP_FILE, output_file=OUTPUT_FILE,
                     use_harmonic=True, use_acapella=True)
    start_time = time.time()
    _upload_file_by_path(file_path=OUTPUT_FILE, bucket=bucket, file_key=filename)
    print('uploaded', time.time() - start_time)
