import argparse

class Exporter(object):

  def copy(self):
    self.select_query = ("SELECT * FROM `{src}`;".format(
      src=self.src
    ))

  def shuffle_sampling(self, column, sample_size):
    self.select_query = ("SELECT {column} FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp}".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

  def shuffle_sampling_with_md5(self, column, sample_size):
    self.select_query = ("SELECT {column}, MD5({column}) AS md5 FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp}".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

  def select_all(self, column):
    self.select_query = ("SELECT {column} FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column}".format(
      column=column, src=self.src
    ))

  def select_all_with_md5(self, column):
    self.select_query = ("SELECT {column}, MD5({column}) AS md5 FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column}".format(
      column=column, src=self.src
    ))

  def select_null_count(self, column):
    self.select_query = ("SELECT bucket_dt, SUM(is_null) AS null_count, COUNT(is_null) AS total_count FROM (SELECT CONCAT(SUBSTR(CAST({bucket} AS STRING), 0, 15), '0') AS bucket_dt, IF({column} IS NULL, 1, 0) AS is_null FROM `{src}`) {tmp} GROUP BY bucket_dt ORDER BY bucket_dt".format(
      column=column, src=self.src, tmp = self.tmp_table, bucket=self.bucket
    ))

class HiveExporter(Exporter):

  def __init__(self, project, src_dataset, src_table, tmp_dataset, tmp_table):
    import pyhs2
    self.project = project
    self.src = "{dataset}.{table}".format(dataset=src_dataset, table=src_table)
    self.tmp_dataset = tmp_dataset
    self.tmp_table = tmp_table
    self.client = pyhs2.connect(host="mst004.bigdl.nmn.io", port=10000, authMechanism="PLAIN", database="gamelog", user="hive", password="***")
    self.select_query = ""
    self.bucket = "i_regdatetime"

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
    self.bucket = "regdatetime"

  def export_to_table(self, table_ref, client, query):
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig()
    job_config.destination = table_ref
    job_config.allow_large_results = True

    # Set configuration.query.createDisposition
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED

    # Set configuration.query.writeDisposition
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Start the query
    job = client.query(query, job_config=job_config)

    # Wait for the query to finish
    job.result()

  def export_to_gcs(self, table_ref, client, dst_uri):
    from google.cloud import bigquery

    job_config = bigquery.ExtractJobConfig()

    # use delimiter which never exists in data
    job_config.field_delimiter = '\t'
    job_config.print_header = False

    extract_job = client.extract_table(
      table_ref,
      dst_uri + "/000000_0", # this naming rule has to be identical to datalake-hive for future useage efficiency
      location='US', # API request
      job_config=job_config)

    extract_job.result()  # Waits for job to complete.

  def export(self, dst_uri):
    # Set resource references
    dataset_ref = self.client.dataset(self.tmp_dataset, project=self.project)
    table_ref = dataset_ref.table(self.tmp_table)

    # create temp table
    self.export_to_table(table_ref, self.client, self.select_query)

    # export
    self.export_to_gcs(table_ref, self.client, dst_uri)

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
  parser.add_argument('--data-type', default='train')
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

  data_type = args.data_type

  # set src type
  if src_type == "hive" :
    exporter = HiveExporter(project, src_dataset, src_table, tmp_dataset, tmp_table)

  elif src_type == "bigquery" :
    exporter = BigqueryExporter(project, src_dataset, src_table, tmp_dataset, tmp_table)

  # set additional options
  if data_type in ["train", "eval", "test"] :
    if sample_size == -1 :
      if gen_md5 : exporter.select_all_with_md5(column)
      else : exporter.select_all(column)
    else :
      if gen_md5 : exporter.shuffle_sampling_with_md5(column, sample_size)
      else : exporter.shuffle_sampling(column, sample_size)
  else :
    if data_type == "rule_null_count_raw" :
      exporter.select_null_count(column)

  exporter.export(dst_uri)
