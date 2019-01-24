from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ConfigParser as cp
import argparse
import tensorflow as tf

from tensorflow.python.tools import saved_model_utils

def get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key):
  return meta_graph_def.signature_def[signature_def_key].inputs

def get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key):
  return meta_graph_def.signature_def[signature_def_key].outputs

def get_input_and_output_names(saved_model_dir, tag_set, signature_def_key):

  meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
  inputs_tensor_info = get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
  outputs_tensor_info = get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)

  inputs = {input_key:input_tensor.name for input_key, input_tensor in inputs_tensor_info.items()}
  outputs = {output_key:output_tensor.name for output_key, output_tensor in outputs_tensor_info.items()}

  return inputs, outputs

def get_context(app_name, project, keyfile, submit_host, python_lib, python_files):

  # generate environment variables
  full_path_list = ['file:{python_lib}/{file}'.format(python_lib=python_lib, file=file) for file in python_files]
  full_paths = ','.join(full_path_list)
  python_path = ':'.join(python_files)

  # cluster execution
  conf = SparkConf() \
    .setMaster(submit_host) \
    .setAppName(app_name) \
    .set('spark.yarn.dist.files','{full_paths}'.format(full_paths=full_paths)) \
    .setExecutorEnv('PYTHONPATH','{python_path}'.format(python_path=python_path)) \

  context = SparkContext(conf=conf)

  # Setup gcs Hadoop Configurations programmatically
  # Require Google Service account
  context._jsc.hadoopConfiguration().set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
  context._jsc.hadoopConfiguration().set("fs.gs.project.id", project)
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.enable", "true")
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile", keyfile)

  return context

def get_list(csv):
  raw_list = csv.split(',')
  stripped_list = [v.strip() for v in raw_list]
  return stripped_list

def get_string(value) :
  if isinstance(value, str) : return value
  else : return str(value)

def inference(it, num_workers, args):
  from tensorflowonspark import util

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.keyfile

  # consume worker number from RDD partition iterator
  for i in it:
    worker_num = i
  print("worker_num: {}".format(i))

  # setup env for single-node TF
  util.single_node_env()

  # load saved_model using default tag and signature
  model_dir = args.model_dir
  tag_set = 'serve'
  signature_def_key = 'serving_default'
  sess = tf.Session()
  tf.saved_model.loader.load(sess, [tag_set], model_dir)

  # define a new tf.data.Dataset (for inferencing)
  ds = tf.data.Dataset.list_files("{}*".format(args.input_dir))
  ds = ds.shard(num_workers, worker_num)
  ds = ds.interleave(tf.data.TextLineDataset, cycle_length=1)
  ds = ds.batch(100)
  iterator = ds.make_one_shot_iterator()
  input = iterator.get_next()

  # create an output file per spark worker for the predictions
  tf.gfile.MakeDirs(args.output_dir)
  output_file = tf.gfile.GFile("{}/prediction.results-{:05d}-of-{:05d}".format(args.output_dir, worker_num, num_workers), mode='w')

  inputs, outputs = get_input_and_output_names(model_dir, tag_set, signature_def_key)

  sorted_outputs = sorted(outputs.items())
  output_keys = [key for key, value in sorted_outputs]
  output_names = [value for key, value in sorted_outputs]

  while True:
    try:
      dataset = sess.run(input)
      result = sess.run(output_names, feed_dict={inputs.values()[0]: dataset})

      cols = len(result)
      rows = len(result[0])
      for row_num in range(rows) :
        row = bytearray("{")
        for col_num in range(cols) :
          if col_num > 0 : row.append(",")
          key = output_keys[col_num]
          value = result[col_num][row_num]
          row += bytearray("'")
          row += bytearray(key.decode('utf-8'), 'utf-8')
          row += bytearray("':")
          if isinstance(value, str) :
            row += bytearray("'" + value.decode('utf-8') + "'", 'utf-8')
          else :
            row += bytearray(str(value))
        row += bytearray("}\n")
        output_file.write(bytes(row))
    except tf.errors.OutOfRangeError:
      break

  output_file.close()


if __name__ == '__main__':
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  config = cp.ConfigParser()
  config.readfp(open('{PROJECT_ROOT}/defaults.cfg'.format(**os.environ)))

  project = config.get('gcp', 'project')
  keyfile = "/etl/credentials/bi-service-155107.json"
  app_name = 'lqad_train_on_spark_test'

  driver = config.get('mysql', 'driver')
  url = config.get('mysql', 'url')
  user = config.get('mysql', 'user')
  password = config.get('mysql', 'password')

  jar_dir = config.get('environment', 'jar_dir')
  submit_host = config.get('environment', 'submit_host')

  python_lib = config.get('environment', 'python_lib')
  python_files = get_list(config.get('environment', 'python_files'))
  sc = get_context(app_name, project, keyfile, submit_host, python_lib, python_files)

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster-size", help="number of nodes in the cluster (for S with labelspark Standalone)", type=int, default=num_executors)
  parser.add_argument("--model-dir", help="gcs path of model", type=str, default="lqad_export")
  parser.add_argument('--input-dir', type=str, help='gcs path of input data')
  parser.add_argument("--output-dir", help="gcs path to save predictions", type=str, default="predictions")

  parser.add_argument("--keyfile", type=str, default=keyfile)

  args, _ = parser.parse_known_args()
  print("args: {}".format(args))

  # Not using TFCluster... just running single-node TF instances on each executor
  nodes = list(range(args.cluster_size))
  nodeRDD = sc.parallelize(list(range(args.cluster_size)), args.cluster_size)
  nodeRDD.foreachPartition(lambda worker_num: inference(worker_num, args.cluster_size, args))