import os
import sys
import ConfigParser as cp
import argparse
from pyspark.sql import SQLContext
from datetime import datetime
from datetime import timedelta

sys.path.insert(0, 'reporter.zip')
from reporter.engines import Reporter

# 95 % : 1.96
# 99 % : 2.58
# 99.7 % : 3
CONFIDENCE_INTERVAL_WEIGHT = 6

def get_date_string(base_dt, offset_date, format='%Y%m%d'):
  return (base_dt + timedelta(days=offset_date)).strftime(format)

class FreqCheckReporter(Reporter):

  def create_freqcheck_raw(self, sqlcontext, src_table, dst_table):
    query_raw = '''
        select
          _c0 as yyyymmdd,
          _c1 as class,
          _c2 as count
        from {table}
      '''.format(table=src_table)

    df = sqlcontext.sql(query_raw)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    return df

  def create_freqcheck_count(self, sqlcontext, src_table, dst_table):
    query_raw = '''
        select
          '{gamecode}' AS gamecode, 
          '{column}' AS column, 
          _c0 as yyyymmdd,
          _c1 as class,
          _c2 as count
        from {table}
      '''.format(gamecode=gamecode, column=column, table=src_table)

    df = sqlcontext.sql(query_raw)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    return df

  def create_freqcheck_memoization(self, sqlcontext, target_dt, src_table, dst_table):

    dst_lowerbound = get_date_string(target_dt, -3, '%Y-%m-%d')
    dst_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')
    src_lowerbound = get_date_string(target_dt, -6, '%Y-%m-%d')
    src_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')

    query = '''
        SELECT T1.P0 as yyyymmdd, T1.class as class, max(T1.xi) as xi, sum(T2.xi) as x FROM
          (SELECT 
              yyyymmdd AS P0, 
              FROM_UNIXTIME(UNIX_TIMESTAMP(yyyymmdd) - 86400*3) AS P0_L, yyyymmdd AS P0_R,
              class, count as xi 
          FROM {table}
          WHERE '{dst_lb}' <= yyyymmdd AND yyyymmdd < '{dst_ub}'
          ) T1
        JOIN
          (SELECT yyyymmdd, class, count as xi 
          FROM {table}
          WHERE '{src_lb}' <= yyyymmdd AND yyyymmdd < '{src_ub}'
          ) T2
        ON T1.class = T2.class
        AND T1.P0_L <= T2.yyyymmdd AND T2.yyyymmdd <= T1.P0_R
        GROUP BY T1.class, T1.P0
      '''.format(gamecode=gamecode, column=column, table=src_table, src_lb=src_lowerbound, src_ub=src_upperbound, dst_lb=dst_lowerbound, dst_ub=dst_upperbound)

    df = sqlcontext.sql(query)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    print('query = ' + query)

    return df

  def create_freqcheck_memoization_2(self, sqlcontext, target_dt, src_table, dst_table):

    query = '''
      SELECT *, (pi-p)/sigma_pi as zi FROM
      (
        SELECT yyyymmdd, class, pi, p, sqrt(p*(1-p)/ni) as sigma_pi FROM
        (
          SELECT T1.yyyymmdd, class, xi, x, ni, n, xi/ni as pi, x/n as p FROM {table}
          JOIN (SELECT yyyymmdd, sum(xi) as ni, sum(x) as n from {table} group by yyyymmdd) T1
          ON {table}.yyyymmdd = T1.yyyymmdd
        ) T2
      ) T3
      '''.format(gamecode=gamecode, column=column, table=src_table)

    df = sqlcontext.sql(query)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    print('query = ' + query)

  def create_freqcheck_threshold(self, sqlcontext, target_dt, src_table, dst_table):

    dst_lowerbound = get_date_string(target_dt, 0, '%Y-%m-%d')
    dst_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')
    src_lowerbound = get_date_string(target_dt, -3, '%Y-%m-%d')
    src_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')

    query = '''
      SELECT 
          '{gamecode}' AS gamecode, 
          '{column}' AS column, 
          yyyymmdd, class,  p-{conf_interval_weight}*sigma_pi*sigma_zi as lowerbound, p+{conf_interval_weight}*sigma_pi*sigma_zi as upperbound FROM
      (
        SELECT T1.P0 as yyyymmdd, T1.class as class, max(pi) as pi, max(p) as p, max(sigma_pi) as sigma_pi, stddev_pop(zi) as sigma_zi FROM
          (SELECT 
            yyyymmdd AS P0, 
            FROM_UNIXTIME(UNIX_TIMESTAMP(yyyymmdd) - 86400*3) AS P0_L, yyyymmdd AS P0_R,
            class, pi, p, sigma_pi
          FROM {table}
          WHERE '{dst_lb}' <= yyyymmdd AND yyyymmdd < '{dst_ub}'
          ) T1
        JOIN
          (SELECT yyyymmdd, class, zi 
          FROM {table}
          WHERE '{src_lb}' <= yyyymmdd AND yyyymmdd < '{src_ub}'
          ) T2
        ON T1.class = T2.class
        AND T1.P0_L <= T2.yyyymmdd AND T2.yyyymmdd <= T1.P0_R
        GROUP BY T1.class, T1.P0
      ) T3
    '''.format(conf_interval_weight=CONFIDENCE_INTERVAL_WEIGHT, gamecode=gamecode, column=column, table=src_table, src_lb=src_lowerbound, src_ub=src_upperbound, dst_lb=dst_lowerbound, dst_ub=dst_upperbound)

    df = sqlcontext.sql(query)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    print('query = ' + query)

    return df

  def compute_threshold(self, gamecode, column, yyyymmdd, temp_table='temp'):

    # base settings
    target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')
    app_name = 'lqad_freqcheck_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    sqlcontext = SQLContext(context, session)

    # extract data from hdfs
    df = self.get_src_df(session, self.src_files, '\t')
    sqlcontext.registerDataFrameAsTable(df, temp_table)

    # initialize raw data for next analysis
    raw_table = temp_table + '_raw'
    self.create_freqcheck_raw(sqlcontext, temp_table, raw_table)

    # memoization for prevention of re-computation
    # generate p, sigma_pi, zi from raw with 3 day window
    memo_table = temp_table + '_memo'
    self.create_freqcheck_memoization(sqlcontext, target_dt, raw_table, memo_table)
    memo_table_2 = temp_table + '_memo_2'
    self.create_freqcheck_memoization_2(sqlcontext, target_dt, memo_table, memo_table_2)

    # generate threshold from memo with 3 day window
    threshold_table = temp_table + '_threshold'
    threshold = self.create_freqcheck_threshold(sqlcontext, target_dt, memo_table_2, threshold_table)

    return threshold

  def compute_freqcount(self, gamecode, column, yyyymmdd, temp_table='temp'):

    # base settings
    target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')
    app_name = 'lqad_freqcheck_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    sqlcontext = SQLContext(context, session)

    # extract data from hdfs
    df = self.get_src_df(session, self.src_files, '\t')
    sqlcontext.registerDataFrameAsTable(df, temp_table)

    # initialize raw data for next analysis
    raw_table = temp_table + '_raw'
    freqcount = self.create_freqcheck_count(sqlcontext, temp_table, raw_table)

    return freqcount

def get_src_files(yyyymmdd, past_days) :
  target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')

  src_files = []
  for i in range(-past_days,1) :
    yyyymmdd = get_date_string(target_dt, i)
    src_files.append("{src_dir}/{yyyymmdd}/rule_freq_count_raw/*".format(src_dir=src_dir, yyyymmdd=yyyymmdd))

  print('files = ' + str(src_files))

  return src_files

if __name__ == "__main__":

  config = cp.ConfigParser()
  config.readfp(open('{PROJECT_ROOT}/defaults.cfg'.format(**os.environ)))

  parser = argparse.ArgumentParser(description='Process data and path info.')
  parser.add_argument('--project', default=config.get('gcp', 'project'), help='')
  parser.add_argument('--src-dir', help='')
  parser.add_argument('--gamecode', help='')
  parser.add_argument('--column', help='')
  parser.add_argument('--yyyymmdd', help='')
  parser.add_argument('--keyfile', default=config.get('gcp', 'keyfile'), help='')
  parser.add_argument('--cmd', default='update_threshold', help='')
  args = parser.parse_args()

  project = args.project
  src_dir = args.src_dir
  gamecode = args.gamecode
  column = args.column
  yyyymmdd = args.yyyymmdd
  keyfile = args.keyfile
  cmd = args.cmd

  if cmd == 'update_threshold' :
    dst_table = 'rulebase_freqcheck_threshold'
    src_files = get_src_files(yyyymmdd, 7)
    reporter = FreqCheckReporter(project, src_files, dst_table, keyfile, config)
    threshold = reporter.compute_threshold(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(threshold)

  elif cmd == 'update_count' :
    dst_table = 'rulebase_freqcheck_count'
    src_files = get_src_files(yyyymmdd, 0)
    reporter = FreqCheckReporter(project, src_files, dst_table, keyfile, config)
    freqcount = reporter.compute_freqcount(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(freqcount)


