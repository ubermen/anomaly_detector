from task_builder import TaskBuilder
from airflow.models import DAG
from datetime import datetime

dag_id = 'nullcheck_sample'

tasks = '''{

init_time:"-1d",
export_nullcount:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh globalsignin i_devicemodel %Y%m%d rule_null_count_raw -1 false"},
update_threshold:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/report_nullcheck.sh update_threshold hdfs://datalake/lqad globalsignin i_devicemodel %Y%m%d", upstream:[export_nullcount]},
update_count:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/report_nullcheck.sh update_count hdfs://datalake/lqad globalsignin i_devicemodel %Y%m%d", upstream:[export_nullcount]}

}'''

default_args = {'start_date': datetime(2019,2,8)}
dag = DAG(dag_id, schedule_interval='0 0 * * *', concurrency=1, max_active_runs=1, default_args=default_args)
TaskBuilder().set(dag, tasks).build_tasks()

