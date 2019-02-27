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
CONFIDENCE_INTERVAL_WEIGHT = 3
ADJACENT_WINDOW_SIZE = 600

def get_date_string(base_dt, offset_date, format='%Y%m%d'):
  return (base_dt + timedelta(days=offset_date)).strftime(format)

class NullCheckReporter(Reporter):

  def create_nullcheck_raw(self, sqlcontext, src_table, dst_table):
    query_raw = '''
        select
          _c0 as bucket_dt,
          _c1 as null_count,
          _c2 as total_count
        from {table}
      '''.format(table=src_table)

    df = sqlcontext.sql(query_raw)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    return df

  def create_nullcheck_count(self, sqlcontext, src_table, dst_table):
    query_raw = '''
        select
          '{gamecode}' AS gamecode, 
          '{column}' AS column, 
          _c0 as bucket_dt,
          _c1 as null_count,
          _c2 as total_count
        from {table}
      '''.format(gamecode=gamecode, column=column, table=src_table)

    df = sqlcontext.sql(query_raw)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    return df

  def create_nullcheck_memoization(self, sqlcontext, target_dt, src_table, dst_table):

    dst_lowerbound = get_date_string(target_dt, -3, '%Y-%m-%d')
    dst_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')
    src_lowerbound = get_date_string(target_dt, -6, '%Y-%m-%d')
    src_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')

    query = '''
        SELECT P0 AS bucket_dt, p, sigma_pi, (pi-p)/sigma_pi AS zi FROM
        (SELECT P0, pi, p, SQRT(p*(1-p)/ni) AS sigma_pi FROM
          (SELECT P0, ni, xi/ni AS pi, sum_x/sum_n AS p FROM
            (SELECT P0, MIN(T2.null_count) AS xi, MIN(T2.total_count) AS ni, SUM(T3.null_count) AS sum_x, SUM(T3.total_count) AS sum_n FROM
              (SELECT 
                  P0, 
                  null_count, total_count, 
                  FROM_UNIXTIME(UNIX_TIMESTAMP(P0) - {adj_window}) AS P0_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P0) + {adj_window}) AS P0_R,
                  FROM_UNIXTIME(UNIX_TIMESTAMP(P1) - {adj_window}) AS P1_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P1) + {adj_window}) AS P1_R,
                  FROM_UNIXTIME(UNIX_TIMESTAMP(P2) - {adj_window}) AS P2_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P2) + {adj_window}) AS P2_R,
                  FROM_UNIXTIME(UNIX_TIMESTAMP(P3) - {adj_window}) AS P3_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P3) + {adj_window}) AS P3_R
                FROM
                (SELECT 
                  null_count, total_count,
                  bucket_dt AS P0, 
                  FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400) AS P1, 
                  FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400*2) AS P2, 
                  FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400*3) AS P3 
                FROM {table} 
                WHERE '{dst_lb}' <= bucket_dt AND bucket_dt < '{dst_ub}'
                ) T1 
              ) T2
            INNER JOIN
              (SELECT bucket_dt, null_count, total_count 
              FROM {table}
              WHERE '{src_lb}' <= bucket_dt AND bucket_dt < '{src_ub}'
              ) T3
            ON T2.P0_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P0_R
            OR T2.P1_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P1_R
            OR T2.P2_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P2_R
            OR T2.P3_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P3_R
            GROUP BY P0 
            ) T4
          ) T5
        ) T6
      '''.format(adj_window=ADJACENT_WINDOW_SIZE, gamecode=gamecode, column=column, table=src_table, src_lb=src_lowerbound, src_ub=src_upperbound, dst_lb=dst_lowerbound, dst_ub=dst_upperbound)

    df = sqlcontext.sql(query)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    print('query = ' + query)

    return df

  def create_nullcheck_threshold(self, sqlcontext, target_dt, src_table, dst_table):

    dst_lowerbound = get_date_string(target_dt, 0, '%Y-%m-%d')
    dst_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')
    src_lowerbound = get_date_string(target_dt, -3, '%Y-%m-%d')
    src_upperbound = get_date_string(target_dt, 1, '%Y-%m-%d')

    query = '''
      SELECT '{gamecode}' AS gamecode, '{column}' AS column, P0 AS bucket_dt, p+{conf_interval_weight}*sigma_pi*sigma_zi AS threshold FROM
      (SELECT P0, MIN(T2.p) AS p, MIN(T2.sigma_pi) AS sigma_pi, STDDEV(T3.zi) AS sigma_zi FROM
        (SELECT 
            P0, 
            p, sigma_pi,
            FROM_UNIXTIME(UNIX_TIMESTAMP(P0) - {adj_window}) AS P0_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P0) + {adj_window}) AS P0_R,
            FROM_UNIXTIME(UNIX_TIMESTAMP(P1) - {adj_window}) AS P1_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P1) + {adj_window}) AS P1_R,
            FROM_UNIXTIME(UNIX_TIMESTAMP(P2) - {adj_window}) AS P2_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P2) + {adj_window}) AS P2_R,
            FROM_UNIXTIME(UNIX_TIMESTAMP(P3) - {adj_window}) AS P3_L, FROM_UNIXTIME(UNIX_TIMESTAMP(P3) + {adj_window}) AS P3_R
          FROM
          (SELECT 
            p, sigma_pi,
            bucket_dt AS P0, 
            FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400) AS P1, 
            FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400*2) AS P2, 
            FROM_UNIXTIME(UNIX_TIMESTAMP(bucket_dt, 'yyyy-MM-dd HH:mm') - 86400*3) AS P3 
          FROM {table} 
          WHERE '{dst_lb}' <= bucket_dt AND bucket_dt < '{dst_ub}'
          ) T1 
        ) T2
      INNER JOIN
        (SELECT bucket_dt, zi 
        FROM {table}
        WHERE '{src_lb}' <= bucket_dt AND bucket_dt < '{src_ub}'
        ) T3
      ON T2.P0_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P0_R
      OR T2.P1_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P1_R
      OR T2.P2_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P2_R
      OR T2.P3_L <= T3.bucket_dt AND T3.bucket_dt <= T2.P3_R
      GROUP BY P0 
      ) T4
    '''.format(conf_interval_weight=CONFIDENCE_INTERVAL_WEIGHT, adj_window=ADJACENT_WINDOW_SIZE, gamecode=gamecode, column=column, table=src_table, src_lb=src_lowerbound, src_ub=src_upperbound, dst_lb=dst_lowerbound, dst_ub=dst_upperbound)

    df = sqlcontext.sql(query)
    sqlcontext.registerDataFrameAsTable(df, dst_table)

    print('query = ' + query)

    return df

  def compute_threshold(self, gamecode, column, yyyymmdd, temp_table='temp'):

    # base settings
    target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')
    app_name = 'lqad_nullcheck_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    sqlcontext = SQLContext(context, session)

    # extract data from hdfs
    df = self.get_src_df(session, self.src_files, '\t')
    sqlcontext.registerDataFrameAsTable(df, temp_table)

    # initialize raw data for next analysis
    raw_table = temp_table + '_raw'
    self.create_nullcheck_raw(sqlcontext, temp_table, raw_table)

    # memoization for prevention of re-computation
    # generate p, sigma_pi, zi from raw with 3 day * 1 hour window
    memo_table = temp_table + '_memo'
    self.create_nullcheck_memoization(sqlcontext, target_dt, raw_table, memo_table)

    # generate threshold from memo with 3 day * 1 hour window
    threshold_table = temp_table + '_threshold'
    threshold = self.create_nullcheck_threshold(sqlcontext, target_dt, memo_table, threshold_table)

    return threshold

  def compute_nullcount(self, gamecode, column, yyyymmdd, temp_table='temp'):

    # base settings
    target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')
    app_name = 'lqad_nullcheck_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
    session = self.get_session(context)
    sqlcontext = SQLContext(context, session)

    # extract data from hdfs
    df = self.get_src_df(session, self.src_files, '\t')
    sqlcontext.registerDataFrameAsTable(df, temp_table)

    # initialize raw data for next analysis
    raw_table = temp_table + '_raw'
    nullcount = self.create_nullcheck_count(sqlcontext, temp_table, raw_table)

    return nullcount

def get_src_files(yyyymmdd, past_days) :
  target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')

  src_files = []
  for i in range(-past_days,1) :
    yyyymmdd = get_date_string(target_dt, i)
    src_files.append("{src_dir}/{yyyymmdd}/rule_null_count_raw/*".format(src_dir=src_dir, yyyymmdd=yyyymmdd))

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
    dst_table = 'rulebase_nullcheck_threshold'
    src_files = get_src_files(yyyymmdd, 7)
    reporter = NullCheckReporter(project, src_files, dst_table, keyfile, config)
    threshold = reporter.compute_threshold(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(threshold)

  elif cmd == 'update_count' :
    dst_table = 'rulebase_nullcheck_count'
    src_files = get_src_files(yyyymmdd, 0)
    reporter = NullCheckReporter(project, src_files, dst_table, keyfile, config)
    nullcount = reporter.compute_nullcount(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(nullcount)


