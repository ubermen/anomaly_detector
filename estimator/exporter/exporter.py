import argparse
from google.cloud import bigquery

class Exporter(object):

  def __init__(self, project, src_dataset, src_table, tmp_dataset, tmp_table):
    self.project = project
    self.dataset = src_dataset
    self.table = src_table
    self.tmp_dataset = tmp_dataset
    self.tmp_table = tmp_table
    self.client = bigquery.Client()

  def export(self, dst_uri):
    dataset_ref = self.client.dataset(self.tmp_dataset, project=self.project)
    table_ref = dataset_ref.table(self.tmp_table)
    job_config = bigquery.ExtractJobConfig()

    job_config.field_delimiter = '\t'
    job_config.print_header = False

    extract_job = self.client.extract_table(
      table_ref,
      dst_uri, # Location must match that of the source table.
      location='US', # API request
      job_config=job_config)

    extract_job.result()  # Waits for job to complete.

  def execute(self, query):
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

  def copy(self):
    query = ("SELECT * FROM `{project}.{dataset}.{table}`;".format(
      project=self.project, dataset=self.dataset, table=self.table))

    self.execute(query)

  def shuffle_sampling(self, column, sample_size):
    query = ("SELECT {column} FROM (SELECT {column}, rand() AS rnum FROM `{project}.{dataset}.{table}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size});".format(
      column=column, sample_size=sample_size, project=self.project, dataset=self.dataset, table=self.table
    ))

    self.execute(query)

  def shuffle_sampling_with_pad(self, column, sample_size, length):
    query = ("SELECT SUBSTR(CONCAT({column}, '\t', CAST(TO_HEX(MD5({column})) AS STRING)), 0, {length}) AS {column} FROM (SELECT {column}, rand() AS rnum FROM `{project}.{dataset}.{table}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size});".format(
      column=column, sample_size=sample_size, length=length, project=self.project, dataset=self.dataset, table=self.table
    ))

    self.execute(query)

  def select_all(self, column):
    query = ("SELECT {column} FROM `{project}.{dataset}.{table}` WHERE {column} IS NOT NULL GROUP BY {column};".format(
      column=column, project=self.project, dataset=self.dataset, table=self.table
    ))

    self.execute(query)

  def select_all_with_pad(self, column, length):
    query = ("SELECT SUBSTR(CONCAT({column}, '\t', CAST(TO_HEX(MD5({column})) AS STRING)), 0, {length}) AS {column} FROM `{project}.{dataset}.{table}` WHERE {column} IS NOT NULL GROUP BY {column};".format(
      column=column, length=length, project=self.project, dataset=self.dataset, table=self.table
    ))

    self.execute(query)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and path info.')
  parser.add_argument('--project', default='bi-service-155107', help='')
  parser.add_argument('--src-dataset', help='')
  parser.add_argument('--src-table', help='')
  parser.add_argument('--tmp-dataset', help='')
  parser.add_argument('--tmp-table', help='')
  parser.add_argument('--column', help='')
  parser.add_argument('--sample-size', default=100000, help='')
  parser.add_argument('--value-length', default=-1, help='')
  parser.add_argument('--dst-uri', help='GCS location from which load data')
  args = parser.parse_args()

  project = args.project

  src_dataset = args.src_dataset
  src_table = args.src_table

  tmp_dataset = args.tmp_dataset
  tmp_table = args.tmp_table

  column = args.column
  sample_size = int(args.sample_size)

  value_length = int(args.value_length)

  dst_uri = args.dst_uri

  exporter = Exporter(project, src_dataset, src_table, tmp_dataset, tmp_table)
  if sample_size == -1 :
    if value_length == -1 : exporter.select_all(column)
    else : exporter.select_all_with_pad(column, value_length)
  else :
    if value_length == -1 : exporter.shuffle_sampling(column, sample_size)
    else : exporter.shuffle_sampling_with_pad(column, sample_size, value_length)

  exporter.export(dst_uri)
