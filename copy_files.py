# pip3 install boto

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_BUCKET_NAME = 'noise-reduction-data'


import boto
import sys
from boto.s3.key import Key


def _connect_to_s3():
    s3_conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return s3_conn


def _get_bucket(bucket=AWS_BUCKET_NAME):
    s3_conn = _connect_to_s3()
    bucket = s3_conn.get_bucket(bucket)
    return bucket


def _upload_file_by_path(file_path, bucket=AWS_BUCKET_NAME, file_key=None):
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    if file_key is None:
        file_key = file_path

    bucket = _get_bucket(bucket)
    key = Key(bucket)
    key.key = file_key
    key.set_contents_from_filename(file_path, cb=percent_cb)


def _get_file_by_key(file_key, file_name=None):
    if file_name is None:
        file_name = file_key

    bucket = _get_bucket()
    key = Key(bucket)
    key.key = file_key
    key.get_contents_to_filename(file_name)


if __name__ == "__main__":
    file_key = '1.wav'
    #_upload_file_by_path(file_name)
    _get_file_by_key(file_key, 'data.wav')