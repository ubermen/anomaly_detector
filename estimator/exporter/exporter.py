import argparse

class Exporter(object):

  def copy(self):
    query = ("SELECT * FROM `{src}`;".format(
      src=self.src
    ))

    self.execute(query)

  def shuffle_sampling(self, column, sample_size):
    query = ("SELECT {column} FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp}".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

    self.execute(query)

  def shuffle_sampling_with_md5(self, column, sample_size):
    query = ("SELECT {column}, MD5({column}) AS md5 FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp}".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

    self.execute(query)

  def select_all(self, column):
    query = ("SELECT {column} FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column};".format(
      column=column, src=self.src
    ))

    self.execute(query)

  def select_all_with_md5(self, column):
    query = ("SELECT {column}, MD5({column}) AS md5 FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column};".format(
      column=column, src=self.src
    ))

    self.execute(query)

class HiveExporter(Exporter):

  def __init__(self, project, src_dataset, src_table, tmp_dataset, tmp_table):
    import pyhs2
    self.project = project
    self.src = "{dataset}.{table}".format(dataset=src_dataset, table=src_table)
    self.tmp_dataset = tmp_dataset
    self.tmp_table = tmp_table
    self.client = pyhs2.connect(host="localhost", port=10000, authMechanism="PLAIN", database="gamelog", user="userid", password="pw")
    self.select_query = ""

  def execute(self, query):
    self.select_query = query

  def export(self, dst_uri):
    export_query = "INSERT OVERWRITE DIRECTORY '{dst_uri}' ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t' {select_query}".format(dst_uri=dst_uri, select_query=self.select_query)
    print(export_query)
    self.client.cursor().execute(export_query)

class BigqueryExporter(Exporter):

  def __init__(self, project, src_dataset, src_table, tmp_dataset, tmp_table):
    from google.cloud import bigquery
    self.project = project
    self.src = "{project}.{dataset}.{table}".format(project=project, dataset=src_dataset, table=src_table)
    self.tmp_dataset = tmp_dataset
    self.tmp_table = tmp_table
    self.client = bigquery.Client()

  def execute(self, query):
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig()

    # Set configuration.query.destinationTable
    destination_dataset = self.client.dataset(self.tmp_dataset)
    destination_table = destination_dataset.table(self.tmp_table)
    job_config.destination = destination_table
    job_config.allow_large_results = True

    # Set configuration.query.createDisposition
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED

    # Set configuration.query.writeDisposition
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Start the query
    job = self.client.query(query, job_config=job_config)

    # Wait for the query to finish
    job.result()

  def export(self, dst_uri):
    from google.cloud import bigquery
    dataset_ref = self.client.dataset(self.tmp_dataset, project=self.project)
    table_ref = dataset_ref.table(self.tmp_table)
    job_config = bigquery.ExtractJobConfig()

    # use delimiter which never exists in data
    job_config.field_delimiter = '\t'
    job_config.print_header = False

    extract_job = self.client.extract_table(
      table_ref,
      dst_uri + "/000000_0", # this naming rule has to be identical to datalake-hive for future useage efficiency
      location='US', # API request
      job_config=job_config)

    extract_job.result()  # Waits for job to complete.

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and path info.')
  parser.add_argument('--project', default='bi-service-155107', help='')
  parser.add_argument('--src-dataset', help='')
  parser.add_argument('--src-table', help='')
  parser.add_argument('--tmp-dataset', help='')
  parser.add_argument('--tmp-table', help='')
  parser.add_argument('--column', help='')
  parser.add_argument('--sample-size', default=100000, help='')
  parser.add_argument('--gen-md5', default=False, type=lambda x: (str(x).lower() == 'true'), help='')
  parser.add_argument('--dst-uri', help='GCS location from which load data')
  parser.add_argument('--src-type', default='hive')
  args = parser.parse_args()

  project = args.project

  src_dataset = args.src_dataset
  src_table = args.src_table

  tmp_dataset = args.tmp_dataset
  tmp_table = args.tmp_table

  column = args.column
  sample_size = int(args.sample_size)

  gen_md5 = args.gen_md5

  dst_uri = args.dst_uri
  src_type = args.src_type

  # set src type
  if src_type == "hive" :
    exporter = HiveExporter(project, src_dataset, src_table, tmp_dataset, tmp_table)

  elif src_type == "bigquery" :
    exporter = BigqueryExporter(project, src_dataset, src_table, tmp_dataset, tmp_table)

  # set additional options
  if sample_size == -1 :
    if gen_md5 : exporter.select_all_with_md5(column)
    else : exporter.select_all(column)
  else :
    if gen_md5 : exporter.shuffle_sampling_with_md5(column, sample_size)
    else : exporter.shuffle_sampling(column, sample_size)

  exporter.export(dst_uri)
