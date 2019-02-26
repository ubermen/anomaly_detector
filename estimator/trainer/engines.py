from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import os
import ConfigParser as cp
import tensorflow as tf
import trainer.utils as utils

def build_input_fns(data_dir, batch_size):

  # Build an iterator over training batches.
  training_dataset = get_dataset_from_file(data_dir, 'train')
  training_dataset = training_dataset.repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the evaluation set.
  eval_dataset = get_dataset_from_file(data_dir, 'eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn

def serving_input_fn():
  string_array = tf.placeholder(tf.string, [None])
  return tf.estimator.export.TensorServingInputReceiver(string_array, string_array)

def get_dataset_from_file(data_dir, data_type):
  dataset = tf.data.Dataset.list_files(data_dir + '/' + data_type + "/*")
  dataset = dataset.interleave(tf.data.TextLineDataset, cycle_length=1)
  return dataset

class Engine(object) :

  def set_graph_modules(self, model_fn):
    self.model_fn = model_fn

  def execute(self, args, ctx=None):

    train_input_fn, eval_input_fn = build_input_fns(args.data_dir, args.batch_size)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.keyfile
    tf.gfile.MakeDirs(args.job_dir)

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.max_steps)

    exporter = tf.estimator.FinalExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=args.viz_steps,
      exporters=[exporter],
      name='lqad-eval')

    run_config = tf.estimator.RunConfig(session_config=utils.get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)

    estimator = tf.estimator.Estimator(
      self.model_fn,
      params=args,
      config=run_config
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

class MLEngine(Engine):

  def run(self, model_fn, args):
    self.set_graph_modules(model_fn)
    self.execute(args)

class SparkCluster(Engine):

  def run(self, model_fn, args):
    from tensorflowonspark import TFCluster

    self.set_graph_modules(model_fn)

    config = cp.ConfigParser()
    config.readfp(open('{PROJECT_ROOT}/defaults.cfg'.format(**os.environ)))

    project = config.get('gcp', 'project')
    keyfile = "/etl/credentials/bi-service-155107.json"
    app_name = args.app_name

    submit_host = config.get('environment', 'submit_host')

    python_lib = config.get('environment', 'python_lib')
    python_files = utils.get_list(config.get('environment', 'python_files'))
    sc = utils.get_context(app_name, project, keyfile, submit_host, python_lib, python_files)

    # tf.app.run()
    cluster = TFCluster.run(
      sc, self.execute, args, args.cluster_size, args.num_ps,
      tensorboard=args.tensorboard,
      input_mode=TFCluster.InputMode.TENSORFLOW,
      log_dir=args.job_dir,
      master_node='master',
      reservation_timeout=1800
    )

    cluster.shutdown()