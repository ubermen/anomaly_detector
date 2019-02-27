from task_builder import TaskBuilder
from airflow.models import DAG
from datetime import datetime

dag_id = 'kde_sample'

tasks = '''{

init_time:"-1d",
export_train:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh bnsmkr completedmainquestcnt %Y%m%d train 10000 false"},
export_eval:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh bnsmkr completedmainquestcnt %Y%m%d eval 100 false", upstream:[export_train]},
train:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/train_on_spark.sh bnsmkr completedmainquestcnt %Y%m%d simple_kde_app 10", upstream:[export_eval]},

export_test:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh bnsmkr completedmainquestcnt %Y%m%d test -1 false", upstream:[train]},
predict:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/predict_on_spark.sh bnsmkr completedmainquestcnt %Y%m%d 10", upstream:[export_test]},
report:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/report.sh hdfs://datalake/lqad bnsmkr completedmainquestcnt %Y%m%d 10 -1", upstream:[predict]}

}'''

default_args = {'start_date': datetime(2019,2,5)}
dag = DAG(dag_id, schedule_interval='0 0 * * *', concurrency=5, max_active_runs=2, default_args=default_args)
TaskBuilder().set(dag, tasks).build_tasks()

