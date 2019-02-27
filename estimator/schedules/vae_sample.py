from task_builder import TaskBuilder
from airflow.models import DAG
from datetime import datetime

dag_id = 'vae_sample'

tasks = '''{

init_time:"-1d",
export_train:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh globalsignin i_devicemodel %Y%m%d train 10000 true"},
export_eval:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh globalsignin i_devicemodel %Y%m%d eval 100 true", upstream:[export_train]},
train:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/train_on_spark.sh globalsignin i_devicemodel %Y%m%d simple_vae_app 10", upstream:[export_eval]},

export_test:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/export_from_hive.sh globalsignin i_devicemodel %Y%m%d test -1 true", upstream:[train]},
predict:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/predict_on_spark.sh globalsignin i_devicemodel %Y%m%d 10", upstream:[export_test]},
report:{command:"ssh web_admin@mst003.bigdv.nmn.io /home/web_admin/log-quality/report.sh hdfs://datalake/lqad globalsignin i_devicemodel %Y%m%d 10 16", upstream:[predict]}

}'''

default_args = {'start_date': datetime(2019,2,1)}
dag = DAG(dag_id, schedule_interval='0 0 * * *', concurrency=5, max_active_runs=2, default_args=default_args)
TaskBuilder().set(dag, tasks).build_tasks()

