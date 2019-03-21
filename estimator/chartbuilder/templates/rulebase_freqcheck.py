database_name = 'lqad'
metrics = ["yyyymmdd", "class", "pi", "lowerbound", "upperbound", "alert"]
viz_type = 'table'
table_sql = '''
select date_format(yyyymmdd, '%Y%m%d') as yyyymmdd, class, pi, lowerbound, upperbound, if(upperbound < pi or lowerbound > pi, 1, 0) as alert from
(
	SELECT t1.yyyymmdd, t1.class, xi, ni, format(xi/ni, 9) as pi, lowerbound, upperbound FROM 
	(select yyyymmdd, class, `count` as xi from lqad.rulebase_freqcheck_count where gamecode='{gamecode}' and `column`='{column}' and yyyymmdd > DATE_SUB(CURDATE(), INTERVAL 4 DAY)) t1
	join 
    (select yyyymmdd, sum(`count`) as ni from lqad.rulebase_freqcheck_count where gamecode='{gamecode}' and `column`='{column}' and yyyymmdd > DATE_SUB(CURDATE(), INTERVAL 4 DAY) group by yyyymmdd) t2
	on t1.yyyymmdd = t2.yyyymmdd
	join 
    (select yyyymmdd, class, lowerbound, upperbound from lqad.rulebase_freqcheck_threshold where gamecode='{gamecode}' and `column`='{column}' and yyyymmdd > DATE_SUB(CURDATE(), INTERVAL 5 DAY)) t3
	on DATE_SUB(t1.yyyymmdd, INTERVAL 1 DAY) = t3.yyyymmdd and t1.class = t3.class
) t4 order by alert desc, yyyymmdd desc, pi desc
'''