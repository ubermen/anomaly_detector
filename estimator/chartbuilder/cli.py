# Add CLI functions to /usr/local/lib/python3.6/site-packages/superset/cli.py
# execution example : superset chartbuilder_create_chart -t rulebase_nullcheck -g globalsignin -c city

@app.cli.command()
@click.option('--type', '-t', help='dashboard type')
@click.option('--gamecode', '-g', help='target gamecode')
@click.option('--column', '-c', help='target column')
def chartbuilder_create_chart(type, gamecode, column):
  from superset.chartbuilder import create_chart
  """ chartbuilder_create_chart """
  tbl_name = '{type}_{gamecode}_{column}'.format(type=type, gamecode=gamecode, column=column)
  print('Loading [ChartBuilder] ' + tbl_name)
  create_chart(type, gamecode, column)