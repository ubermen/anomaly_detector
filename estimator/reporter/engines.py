from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

class Reporter(object):

  def __init__(self, project, src_files, dst_table, keyfile, config):

    self.project = project
    self.src_files = src_files
    self.dst_table = dst_table
    self.keyfile = keyfile

    self.driver = config.get('mysql', 'driver')
    self.url = config.get('mysql', 'url')
    self.user = config.get('mysql', 'user')
    self.password = config.get('mysql', 'password')

    self.jar_dir = config.get('environment', 'jar_dir')
    self.submit_host = config.get('environment', 'submit_host')

    self.python_lib = config.get('environment', 'python_lib')
    self.python_files = self.get_list(config.get('environment', 'python_files'))

  def get_list(self, csv):
    raw_list = csv.split(',')
    stripped_list = [v.strip() for v in raw_list]
    return stripped_list

  def get_context(self, app_name, project, keyfile, submit_host, python_lib, python_files):

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

  def get_session(self, context):

    session = SparkSession.builder \
      .config(conf=context.getConf()) \
      .getOrCreate()

    return session

  def get_src_df(self, session, src_files, type='json'):

    if type == 'json' :
      df = session.read.format("json") \
        .option("header", "false") \
        .option("inferSchema", "true") \
        .load(src_files)
    else :
      df = session.read.format("csv") \
        .option("delimiter", type) \
        .option("header", "false") \
        .load(src_files)

    return df

  def write_df_to_mysql(self, df):
    df.write.format('jdbc').options(
      url=self.url,
      driver=self.driver,
      dbtable=self.dst_table,
      user=self.user,
      password=self.password).mode('append').save()

