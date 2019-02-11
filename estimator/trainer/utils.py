from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# Dependency imports
import tensorflow as tf

def get_session_config_from_env_var():

  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

  if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']):
    # Master should only communicate with itself and ps
    if tf_config['task']['type'] == 'master':
      return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
    # Worker should only communicate with itself and ps
    elif tf_config['task']['type'] == 'worker':
      return tf.ConfigProto(device_filters=[
        '/job:ps',
        '/job:worker/task:%d' % tf_config['task']['index']
      ])
  return None

def get_list(csv):
  raw_list = csv.split(',')
  stripped_list = [v.strip() for v in raw_list]
  return stripped_list

def get_context(app_name, project, keyfile, submit_host, python_lib, python_files):
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  # generate environment variables
  full_path_list = ['file:{python_lib}/{file}'.format(python_lib=python_lib, file=file) for file in python_files]
  full_paths = ','.join(full_path_list)
  python_path = ':'.join(python_files)

  # cluster execution
  conf = SparkConf() \
    .setMaster(submit_host) \
    .setAppName(app_name) \
    .set('spark.yarn.dist.files','{full_paths}'.format(full_paths=full_paths)) \
    .setExecutorEnv('PYTHONPATH','{python_path}'.format(python_path=python_path))

  context = SparkContext(conf=conf)

  # Setup gcs Hadoop Configurations programmatically
  # Require Google Service account
  context._jsc.hadoopConfiguration().set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
  context._jsc.hadoopConfiguration().set("fs.gs.project.id", project)
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.enable", "true")
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile", keyfile)

  return context