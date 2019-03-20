from superset import db
from superset.connectors.connector_registry import ConnectorRegistry
from superset.connectors.sqla.models import TableColumn
from superset.models.core import Slice, Dashboard, Database

from flask_appbuilder import Model
from sqlalchemy import Column, Integer, ForeignKey

import importlib

# default settings
dashboard_name_template = '{type}_{gamecode}'
chart_name_template = dashboard_name_template + '_{column}'

class DashboardSlices(Model):
  __tablename__ = 'dashboard_slices'
  id = Column(Integer, primary_key=True)
  dashboard_id = Column(Integer, ForeignKey('dashboards.id'))
  slice_id = Column(Integer, ForeignKey('slices.id'))

def create_dashboard(type, gamecode):
  dashboard_name = dashboard_name_template.format(type=type, gamecode=gamecode)
  dashboard = db.session.query(Dashboard).filter_by(dashboard_title=dashboard_name).first()
  if not dashboard:
    dashboard = Dashboard(dashboard_title=dashboard_name)
    db.session.merge(dashboard)
    dashboard = db.session.query(Dashboard).filter_by(dashboard_title=dashboard_name).first()

  TBL = ConnectorRegistry.sources['table']
  tables = db.session.query(TBL).filter(TBL.table_name.like(dashboard_name+'%')).all()

  position_json = '{'
  tname_list = []

  for table in tables :
    id = table.id
    table_name = table.table_name
    column_idx = len(dashboard_name)+1
    column = table_name[column_idx:]

    slice = db.session.query(Slice).filter_by(datasource_id=id).first()
    slice_json = '''
        "CHART-{tname}":{{"id":"CHART-{tname}", "meta":{{"sliceName":"{tname}", "chartId":{id}, "height":52, "width":12}}, "type":"CHART"}},
        "ROW-{tname}":{{"id":"ROW-{tname}", "meta":{{"background":"BACKGROUND_TRANSPARENT"}}, "children":["CHART-{tname}"], "type":"ROW"}},
        "TAB-{tname}":{{"id":"TAB-{tname}", "meta":{{"text":"{colname}"}}, "children":["ROW-{tname}"], "type":"TAB"}},
        '''.format(tname=table_name, id=slice.id, colname=column)
    position_json += slice_json
    tname_list.append(table_name)

    dashboard_slices_row = db.session.query(DashboardSlices).filter_by(dashboard_id=dashboard.id, slice_id=slice.id).first()
    if not dashboard_slices_row: dashboard_slices_row = DashboardSlices(dashboard_id=dashboard.id, slice_id=slice.id)
    db.session.merge(dashboard_slices_row)

  count = 0
  unit = 10
  tabs_map = {}
  for tname in tname_list :
    tabs_num = int(count/unit)
    if count % unit == 0 : tabs_map[tabs_num] = []
    tabs_map[tabs_num].append('"TAB-{tname}"'.format(tname=tname))
    count += 1

  tabs_id_list = []
  for tabs_num in tabs_map :
    tabs_id = '"TABS-{tabs_num}"'.format(tabs_num=str(tabs_num))
    tabs_json = '{tabs_id}:{{"id":{tabs_id}, "children":[{tab_id_csv}], "type":"TABS"}},\n'.format(tabs_id=tabs_id, tab_id_csv=','.join(tabs_map[tabs_num]))
    position_json += tabs_json
    tabs_id_list.append(tabs_id)

  grid_json = '"GRID_ID": {{"id":"GRID_ID", "children":[{tabs_id_csv}], "type":"GRID"}},\n'.format(tabs_id_csv=','.join(tabs_id_list))
  position_json += grid_json
  position_json += '''
    "ROOT_ID": {{ "id":"ROOT_ID", "children":["GRID_ID"], "type":"ROOT"}},
    "HEADER_ID": {{"id":"HEADER_ID", "meta":{{ "text":"{dashboard_name}"}}, "type":"HEADER"}},
    "DASHBOARD_VERSION_KEY":"v2"
    }}
    '''.format(dashboard_name=dashboard_name)

  dashboard.position_json = position_json

  db.session.merge(dashboard)
  db.session.commit()

def create_chart(type, gamecode, column, template):
  template_module = importlib.import_module("superset.chartbuilder.templates."+template)

  TBL = ConnectorRegistry.sources['table']
  tbl_name = chart_name_template.format(type=type, gamecode=gamecode, column=column)

  print('Creating table {} reference'.format(tbl_name))
  tbl = db.session.query(TBL).filter_by(table_name=tbl_name).first()
  if not tbl: tbl = TBL(table_name=tbl_name)
  tbl.database_id = db.session.query(Database).filter_by(database_name='lqad').first().id
  tbl.sql = template_module.template.format(gamecode=gamecode, column=column)
  db.session.merge(tbl)

  tbl = db.session.query(TBL).filter_by(table_name=tbl_name).first()
  bucket_dt = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name=template_module.time_column).first()
  if not bucket_dt: bucket_dt = TableColumn(table_id=tbl.id, column_name=template_module.time_column, is_dttm=1)
  db.session.merge(bucket_dt)

  metric_json = []
  for metric in template_module.metrics :
    metric_obj = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name=metric).first()
    if not metric_obj:
      metric_obj = TableColumn(table_id=tbl.id, column_name=metric)
      db.session.merge(metric_obj)
      metric_obj = db.session.query(TableColumn).filter_by(table_id=tbl.id, column_name=metric).first()
    metric_json.append('{{"column": {{"id": {metric_id}, "column_name": "{metric}"}}, "label": "{metric}", "aggregate": "MAX", "expressionType": "SIMPLE"}}' \
                       .format(metric_id=metric_obj.id, metric=metric))

  slice = db.session.query(Slice).filter_by(datasource_id=tbl.id).first()
  if not slice: slice = Slice(datasource_id=tbl.id, slice_name=tbl_name, datasource_name=tbl_name, datasource_type='table', viz_type='line', created_by_fk=1)
  slice.params='''
    {{"datasource": "{datasource}", "granularity_sqla": "{time_column}", "time_grain_sqla": "PT1M", "time_range": "Last week", 
    "metrics": [
    {metric_json}
    ]}}
    '''.format(datasource=str(tbl.id)+'__table', time_column=template_module.time_column, metric_json=',\n'.join(metric_json))
  db.session.merge(slice)

  db.session.commit()