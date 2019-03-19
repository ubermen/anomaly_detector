import os
import sys
from pyspark.sql import SQLContext

sys.path.insert(0, 'reporter.zip')
from reporter.engines import Reporter
from reporter.rule_utils import ruleUtils

class BaseCheckerReporter(Reporter):
    def get_rules(self, type):
        return ['0']

    def get_query(self, src_table, gamecode, column, yyyymmdd, rules):
        utils = ruleUtils()
        regdatetime = utils.get_regdatetime()
        return '''select '{gamecode}' as gamecode, '{column}' as column, '{yyyymmdd}' as yyyymmdd, _c0 as data, '{rules}' as rules, '{regdatetime}' as regdatetime from {table}'''.format(table=src_table, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, rules=rules, regdatetime=regdatetime)


    def check_rule(self, sqlcontext, rule_name, src_table, gamecode, column, yyyymmdd, dst_table):
        rules = self.get_rules(rule_name)
        query_raw = self.get_query(src_table, gamecode, column, yyyymmdd, rules)

        print(query_raw)

        df = sqlcontext.sql(query_raw)
        sqlcontext.registerDataFrameAsTable(df, dst_table)

        return df

    def compute_rule(self, rule_name, gamecode, column, yyyymmdd, temp_table='temp'):

        # base settings
        app_name = 'lqad_' + rule_name + '_{gamecode}_{column}_{yyyymmdd}'.format(gamecode=gamecode, column=column,
                                                                        yyyymmdd=yyyymmdd)
        context = self.get_context(app_name, self.project, self.keyfile, self.submit_host, self.python_lib, self.python_files)
        session = self.get_session(context)
        sqlcontext = SQLContext(context, session)

        # extract data from hdfs
        df = self.get_src_df(session, self.src_files, '\t')
        sqlcontext.registerDataFrameAsTable(df, temp_table)

        # initialize raw data for next analysis
        raw_table = temp_table + '_raw'
        result_count = self.check_rule(sqlcontext, rule_name, temp_table, gamecode, column, yyyymmdd, raw_table)

        return result_count

