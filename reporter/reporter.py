import os
import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

class Reporter(object):

  def __init__(self, project, src_dir, dst_table, keyfile):

    self.project = project
    self.src_dir = src_dir
    self.dst_table = dst_table
    self.keyfile = keyfile

  def get_context(self, project, keyfile):

    # environment settings
    os.environ['PYSPARK_SUBMIT_ARGS'] = """--jars gcs-connector-latest-hadoop2.jar pyspark-shell"""

    conf = SparkConf() \
      .setMaster("local[8]") \
      .setAppName("Test")

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

  def summary(self, sample_count, sample_length, temp_table='temp', sample_delimiter='|'):

    context = self.get_context(self.project, self.keyfile)
    session = self.get_session(context)
    df = self.get_src_df(session, self.src_dir)

    sqlcontext = SQLContext(context, session)
    sqlcontext.registerDataFrameAsTable(df, temp_table)
    sqlcontext.udf.register("sampled_concat", lambda list : sample_delimiter.join(list[:sample_count]))

    if sample_length > 0 :
      query = '''
          select
            floor(_1) as rank,
            count(*) as count,
            sampled_concat(collect_list(substring(_0, 0, {len}))) as sample
          from {table}
          group by floor(_1)
          order by rank
        '''.format(table=temp_table, len=sample_length)
    else :
      query = '''
          select
            floor(_1) as rank,
            count(*) as count,
            sampled_concat(collect_list(_0)) as sample
          from {table}
          group by floor(_1)
          order by rank
        '''.format(table=temp_table)

    summary = sqlcontext.sql(query)

    return summary

# execution example :
# python reporter.py --src-dir gs://bigus/data/globalsignin_devicemodel_prediction --sample_count 3 --sample_length 16
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and path info.')
  parser.add_argument('--project', default='bi-service-155107', help='')
  parser.add_argument('--src-dir', help='')
  parser.add_argument('--dst-table', help='')
  parser.add_argument('--keyfile', default='/home/web_admin/credentials/bi-service.json', help='')
  parser.add_argument('--sample_count', default=1, help='')
  parser.add_argument('--sample_length', default=-1, help='')
  args = parser.parse_args()

  project = args.project
  src_dir = args.src_dir
  dst_table = args.dst_table
  keyfile = args.keyfile
  sample_count = int(args.sample_count)
  sample_length = int(args.sample_length)

  reporter = Reporter(project, src_dir, dst_table, keyfile)
  summary = reporter.summary(sample_count=sample_count, sample_length=sample_length)
  summary.show(200, False)

