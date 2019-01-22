from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import argparse
from google.cloud import storage

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and model path info.')
  parser.add_argument('--dir', help='GCS location to delete')
  args = parser.parse_args()

  # gs://bucket/blob_prefix
  dir = args.dir
  dir = dir[5:]
  cut_pos = dir.index('/')

  bucket = dir[:cut_pos]
  blob_prefix = dir[cut_pos+1:]

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket)
  blobs = bucket.list_blobs(prefix=blob_prefix)
  for blob in blobs:
    blob.delete()