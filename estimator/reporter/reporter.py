import os
import ConfigParser as cp
import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

class Reporter(object):

  def __init__(self, project, src_dir, dst_table, keyfile, config):

    self.project = project
    self.src_dir = src_dir
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

  def get_src_df(self, session, src_dir):

    df = session.read.format("json") \
      .option("header", "false") \
      .option("inferSchema", "true") \
      .load("{src_dir}/prediction.results-*".format(src_dir=src_dir))

    return df

  def summary(self, gamecode, column, yyyymmdd, sample_count, sample_length, temp_table='temp', sample_delimiter='|'):

    app_name = 'lqad_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    df = self.get_src_df(session, self.src_dir)

    sqlcontext = SQLContext(context, session)
    sqlcontext.registerDataFrameAsTable(df, temp_table)
    sqlcontext.udf.register("sampled_concat", lambda list : sample_delimiter.join(list[:sample_count]))

    if sample_length > 0 :
      query = '''
          select
            '{gamecode}' as gamecode,
            '{column}' as column,
            '{yyyymmdd}' as yyyymmdd,
            floor(_1) as rank,
            count(*) as count,
            sampled_concat(collect_list(substring(_0, 0, {len}))) as sample
          from {table}
          group by floor(_1)
          order by rank
        '''.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, table=temp_table, len=sample_length)
    else :
      query = '''
          select
            '{gamecode}' as gamecode,
            '{column}' as column,
            '{yyyymmdd}' as yyyymmdd,
            floor(_1) as rank,
            count(*) as count,
            sampled_concat(collect_list(_0)) as sample
          from {table}
          group by floor(_1)
          order by rank
        '''.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, table=temp_table)

    summary = sqlcontext.sql(query)

    return summary

  def write_df_to_mysql(self, df):
    df.write.format('jdbc').options(
      url=self.url,
      driver=self.driver,
      dbtable=self.dst_table,
      user=self.user,
      password=self.password).mode('append').save()



if __name__ == "__main__":
  # execution example :
  if False : """
  python reporter.py \
  --src-dir gs://bigus/data/globalsignin_devicemodel_prediction \
  --dst-table inherent_anomaly_summary \
  --gamecode globalsignin \
  --column devicemodel \
  --yyyymmdd 20181004 \
  --sample_count 10 \
  --sample_length 16
  """

  config = cp.ConfigParser()
  config.readfp(open('{PROJECT_ROOT}/defaults.cfg'.format(**os.environ)))

  parser = argparse.ArgumentParser(description='Process data and path info.')
  parser.add_argument('--project', default=config.get('gcp', 'project'), help='')
  parser.add_argument('--src-dir', help='')
  parser.add_argument('--dst-table', help='')
  parser.add_argument('--gamecode', help='')
  parser.add_argument('--column', help='')
  parser.add_argument('--yyyymmdd', help='')
  parser.add_argument('--keyfile', default=config.get('gcp', 'keyfile'), help='')
  parser.add_argument('--sample_count', default=1, help='')
  parser.add_argument('--sample_length', default=-1, help='')
  parser.add_argument('--sample_delimiter', default='_|_', help='')
  args = parser.parse_args()

  project = args.project
  src_dir = args.src_dir
  dst_table = args.dst_table
  gamecode = args.gamecode
  column = args.column
  yyyymmdd = args.yyyymmdd
  keyfile = args.keyfile
  sample_count = int(args.sample_count)
  sample_length = int(args.sample_length)
  sample_delimiter = args.sample_delimiter

  reporter = Reporter(project, src_dir, dst_table, keyfile, config)

  summary = reporter.summary(
    gamecode=gamecode, column=column, yyyymmdd=yyyymmdd,
    sample_count=sample_count, sample_length=sample_length, sample_delimiter=sample_delimiter
  )

  # summary.show(200, False)
  reporter.write_df_to_mysql(summary)

