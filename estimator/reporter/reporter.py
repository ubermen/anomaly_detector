import os
import sys
import ConfigParser as cp
import argparse
from pyspark.sql import SQLContext
from datetime import datetime as dt

sys.path.insert(0, 'reporter.zip')
from reporter.engines import Reporter

class SummaryReporter(Reporter):

  def summary(self, gamecode, column, yyyymmdd, sample_count, sample_length, temp_table='temp', sample_delimiter='_|_', regdatetime=dt.now().strftime('%Y-%m-%d %H:%M:%S')):

    app_name = 'lqad_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    df = self.get_src_df(session, self.src_files)

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
            sampled_concat(collect_list(substring(_0, 0, {len}))) as sample,
            '{regdatetime}' as regdatetime
          from {table}
          group by floor(_1)
          order by rank
        '''.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, table=temp_table, len=sample_length, regdatetime=regdatetime)
    else :
      query = '''
          select
            '{gamecode}' as gamecode,
            '{column}' as column,
            '{yyyymmdd}' as yyyymmdd,
            floor(_1) as rank,
            count(*) as count,
            sampled_concat(collect_list(_0)) as sample,
            '{regdatetime}' as regdatetime
          from {table}
          group by floor(_1)
          order by rank
        '''.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, table=temp_table, regdatetime=regdatetime)

    summary = sqlcontext.sql(query)

    return summary

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

  src_files = ["{src_dir}/prediction.results-*".format(src_dir=src_dir)]
  reporter = SummaryReporter(project, src_files, dst_table, keyfile, config)

  summary = reporter.summary(
    gamecode=gamecode, column=column, yyyymmdd=yyyymmdd,
    sample_count=sample_count, sample_length=sample_length, sample_delimiter=sample_delimiter
  )

  # summary.show(200, False)
  reporter.write_df_to_mysql(summary)

