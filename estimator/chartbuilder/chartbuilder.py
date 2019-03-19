from superset import db
from superset.connectors.connector_registry import ConnectorRegistry
from superset.connectors.sqla.models import TableColumn
from superset.models.core import Slice

def create_chart(type, gamecode, column):

  TBL = ConnectorRegistry.sources['table']
  tbl_name = '{type}_{gamecode}_{column}'.format(type=type, gamecode=gamecode, column=column)

  print('Creating table {} reference'.format(tbl_name))
  tbl = db.session.query(TBL).filter_by(table_name=tbl_name).first()
  if not tbl: tbl = TBL(table_name=tbl_name)
  tbl.database_id = 2
  tbl.sql = '''
    SELECT *, IF(threshold < score, -score, null) AS alert FROM
    (
        SELECT 
            T2.bucket_dt, 
            if(T1.threshold is null, 0, T1.threshold) as threshold, 
            T2.score
        FROM
            (SELECT bucket_dt, max(threshold) as threshold FROM lqad.rulebase_nullcheck_threshold WHERE gamecode = '{{ gamecode }}' AND `column` = '{{ column }}' GROUP BY bucket_dt) T1 JOIN
            (SELECT bucket_dt, max(format(null_count/total_count, 9)) AS score FROM lqad.rulebase_nullcheck_count WHERE gamecode = '{{ gamecode }}' AND `column` = '{{ column }}' GROUP BY bucket_dt) T2
            ON T1.bucket_dt = DATE_SUB(T2.bucket_dt, INTERVAL 1 DAY)
    ) T3
    '''

  tbl.template_params = '{{"gamecode":"{gamecode}", "column":"i_{column}"}}'.format(gamecode=gamecode, column=column)
  db.session.merge(tbl)

  tbl = db.session.query(TBL).filter_by(table_name=tbl_name).first()

  bucket_dt = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='bucket_dt').first()
  threshold = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='threshold').first()
  score = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='score').first()
  alert = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='alert').first()
  if not bucket_dt: bucket_dt = TableColumn(table_id=tbl.id, column_name='bucket_dt', is_dttm=1)
  if not threshold: threshold = TableColumn(table_id=tbl.id, column_name='threshold')
  if not score: score = TableColumn(table_id=tbl.id, column_name='score')
  if not alert: alert = TableColumn(table_id=tbl.id, column_name='alert')

  db.session.merge(bucket_dt)
  db.session.merge(threshold)
  db.session.merge(score)
  db.session.merge(alert)

  threshold = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='threshold').first()
  score = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='score').first()
  alert = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name='alert').first()

  slice = db.session.query(Slice).filter_by(datasource_id=tbl.id).first()
  if not slice: slice = Slice(datasource_id=tbl.id, slice_name=tbl_name, datasource_name=tbl_name, datasource_type='table', viz_type='line', created_by_fk=1)
  slice.params='''
    {{"datasource": "{datasource}", "granularity_sqla": "bucket_dt", "time_grain_sqla": "PT1M", "time_range": "Last week", 
    "metrics": [
    {{"column": {{"id": {threshold_id}, "column_name": "threshold"}}, "label": "threshold", "aggregate": "MAX", "expressionType": "SIMPLE"}}, 
    {{"column": {{"id": {score_id}, "column_name": "score"}}, "label": "score", "aggregate": "MAX", "expressionType": "SIMPLE"}}, 
    {{"column": {{"id": {alert_id}, "column_name": "alert"}}, "label": "alert", "aggregate": "MIN", "expressionType": "SIMPLE"}}
    ]}}
    '''.format(datasource=str(tbl.id)+'__table', threshold_id=threshold.id, score_id=score.id, alert_id=alert.id)
  db.session.merge(slice)

  db.session.commit()