import os
import sys
import ConfigParser as cp
import argparse

sys.path.insert(0, 'reporter.zip')
from reporter.base_checker import BaseCheckerReporter
from reporter.rule_utils import ruleUtils

utils = ruleUtils()

class MinMaxCheckReporter(BaseCheckerReporter):
    def get_rules(self, type):
        return ['10' , '100']

    def get_query(self, src_table, gamecode, column, yyyymmdd, rules):
        regdatetime = utils.get_regdatetime()

        min = rules[0]
        max = rules[1]

        if min == '':
            if max == '':
                return '';
            return '''select '{gamecode}' as gamecode, '{column}' as column, '{yyyymmdd}' as yyyymmdd, _c0 as data, '{min}' as min, '{max}' as max, '{regdatetime}' as regdatetime from {table} where _c0 > {max}'''.format(table=src_table, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, min=min, max=max, regdatetime=regdatetime)
        if max == '':
            return '''select '{gamecode}' as gamecode, '{column}' as column, '{yyyymmdd}' as yyyymmdd, _c0 as data, '{min}' as min, '{max}' as max, '{regdatetime}' as regdatetime from {table} where _c0 < {min}'''.format(table=src_table, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, min=min, max=max, regdatetime=regdatetime)

        return '''select '{gamecode}' as gamecode, '{column}' as column, '{yyyymmdd}' as yyyymmdd, _c0 as data, '{min}' as min, '{max}' as max, '{regdatetime}' as regdatetime from {table} where _c0 < {min} or _c0 > {max}'''.format(table=src_table, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd, min=min, max=max, regdatetime=regdatetime)


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

    rule_name = 'min_max'
    dst_table = 'rulebase_' + rule_name
    src_files = utils.get_src_files(src_dir, yyyymmdd, 0)
    reporter = MinMaxCheckReporter(project, src_files, dst_table, keyfile, config)
    min_max_checker = reporter.compute_rule(rule_name=rule_name, gamecode=gamecode, column=column, yyyymmdd=yyyymmdd)
    reporter.write_df_to_mysql(min_max_checker)
