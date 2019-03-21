database_name = 'lqad'
time_column = 'bucket_dt'
metrics = ['threshold', 'score', 'alert']
viz_type = 'line'
table_sql = '''
SELECT *, IF(threshold < score, -score, null) AS alert FROM
(
    SELECT 
        T2.bucket_dt, 
        if(T1.threshold is null, 0, T1.threshold) as threshold, 
        T2.score
    FROM
        (SELECT bucket_dt, max(threshold) as threshold FROM rulebase_nullcheck_threshold WHERE gamecode = '{gamecode}' AND `column` = '{column}' GROUP BY bucket_dt) T1 JOIN
        (SELECT bucket_dt, max(format(null_count/total_count, 9)) AS score FROM rulebase_nullcheck_count WHERE gamecode = '{gamecode}' AND `column` = '{column}' GROUP BY bucket_dt) T2
        ON T1.bucket_dt = DATE_SUB(T2.bucket_dt, INTERVAL 1 DAY)
) T3
'''