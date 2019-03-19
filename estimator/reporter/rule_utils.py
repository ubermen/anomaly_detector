
# import mysql.connector
from datetime import datetime as dt
from datetime import timedelta

class ruleUtils():
    def get_schema(self, type, game_code, logKey):
        # db = mysql.connector.connect(
        #     host="10.109.25.194",
        #     port="58306",
        #     user="QAWeb_ACC",
        #     passwd="wkf@ro$qkf5",
        #     database="NetmarblesLog"
        # )
        #
        # cursor = db.cursor()
        # cursor.execute(
        #     "select " + type + " from TB_GameLogInfoDetail where gameCode = '" + game_code + "' and logkey = '" + logKey + "' limit 1")
        #
        # query_response = cursor.fetchall()
        # data = query_response.pop(0)

        return '10'
        # return '' if not all(data) else ''.join(data)

    def get_regdatetime(self, format='%Y-%m-%d %H:%M:%S'):
        return dt.now().strftime(format)


    def get_date_string(self, base_dt, offset_date, format='%Y%m%d'):
        return (base_dt + timedelta(days=offset_date)).strftime(format)


    def get_src_files(self, src_dir, yyyymmdd, past_days):
        target_dt = dt.strptime(yyyymmdd, '%Y%m%d')

        src_files = []
        for i in range(-past_days, 1):
            yyyymmdd = self.get_date_string(target_dt, i)
            src_files.append("{src_dir}/{yyyymmdd}/test/*".format(src_dir=src_dir, yyyymmdd=yyyymmdd))

        print('files = ' + str(src_files))

        return src_files