import os
import sys
import ConfigParser as cp
import argparse
from pyspark.sql import SQLContext
from datetime import datetime
from datetime import timedelta

sys.path.insert(0, 'reporter.zip')
from reporter.engines import Reporter
from reporter.rule_utils import ruleUtils

utils = ruleUtils()

def get_date_string(base_dt, offset_date, format='%Y%m%d'):
    return (base_dt + timedelta(days=offset_date)).strftime(format)


class TypeCheckReporter(Reporter):

    def get_query(self, src_table, gamecode, column, yyyymmdd, range):
        regdatetime = utils.get_regdatetime()
        print(range)

        return '''select '{gamecode}' as gamecode, '{column}' as column, '{yyyymmdd}' as yyyymmdd, _c0 as data, '{range}' as range, '{regdatetime}' as regdatetime from {table} where length(_c0) <= {range}'''.format(
            table=src_table, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, range=range, regdatetime=regdatetime)

    def type_check(self, sqlcontext, src_table, gamecode, column, yyyymmdd, dst_table, range):
        query_raw = self.get_query(src_table, gamecode, column, yyyymmdd, range)

        print(query_raw)

        df = sqlcontext.sql(query_raw)
        sqlcontext.registerDataFrameAsTable(df, dst_table)

        return df

    def compute_type(self, gamecode, column, yyyymmdd, temp_table='temp'):

        # base settings
        app_name = 'lqad_type_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column,
                                                                        yyyymmdd=yyyymmdd)
        context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
        session = self.get_session(context)
        sqlcontext = SQLContext(context, session)

        range = '2'

        # extract data from hdfs
        df = self.get_src_df(session, self.src_files, '\t')
        sqlcontext.registerDataFrameAsTable(df, temp_table)

        # initialize raw data for next analysis
        raw_table = temp_table + '_raw'
        result_count = self.type_check(sqlcontext, temp_table, gamecode, column, yyyymmdd, raw_table, range)

        return result_count


def get_src_files(yyyymmdd, past_days):
    target_dt = datetime.strptime(yyyymmdd, '%Y%m%d')

    src_files = []
    for i in range(-past_days, 1):
        yyyymmdd = get_date_string(target_dt, i)
        src_files.append("{src_dir}/{yyyymmdd}/test/*".format(src_dir=src_dir, yyyymmdd=yyyymmdd))

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

    dst_table = 'rulebase_type'
    src_files = get_src_files(yyyymmdd, 0)
    reporter = TypeCheckReporter(project, src_files, dst_table, keyfile, config)
    type_checker = reporter.compute_type(gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(type_checker)
