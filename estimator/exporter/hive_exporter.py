import argparse
import pyhs2

class HiveExporter(object):

  def __init__(self, project, src_dataset, src_table, tmp_dataset, tmp_table):
    self.project = project
    self.src = "{dataset}.{table}".format(dataset=src_dataset, table=src_table)
    self.tmp_dataset = tmp_dataset
    self.tmp_table = tmp_table
    self.client = pyhs2.connect(host="host", port=10000, authMechanism="PLAIN", database="db", user="id", password="pw")
    self.select_query = ""

  def export(self, dst_uri):
    export_query = "INSERT OVERWRITE DIRECTORY '{dst_uri}' ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t' {select_query}".format(dst_uri=dst_uri, select_query=self.select_query)
    print(export_query)
    self.client.cursor().execute(export_query)

  def execute(self, query):
    self.select_query = query

  def copy(self):
    query = ("SELECT * FROM `{src}`;".format(
      src=self.src
    ))

    self.execute(query)

  def shuffle_sampling(self, column, sample_size):
    query = ("SELECT {column} FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp})".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

    self.execute(query)

  def shuffle_sampling_with_md5(self, column, sample_size):
    query = ("SELECT {column}, CAST(MD5({column}) AS STRING) AS md5 FROM (SELECT {column}, rand() AS rnum FROM `{src}` WHERE {column} IS NOT NULL ORDER BY rnum LIMIT {sample_size}) {tmp}".format(
      column=column, sample_size=sample_size, src=self.src, tmp = self.tmp_table
    ))

    self.execute(query)

  def select_all(self, column):
    query = ("SELECT {column} FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column}".format(
      column=column, src=self.src
    ))

    self.execute(query)

  def select_all_with_md5(self, column):
    query = ("SELECT {column}, CAST(MD5({column}) AS STRING) AS md5 FROM `{src}` WHERE {column} IS NOT NULL GROUP BY {column}".format(
      column=column, src=self.src
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
  parser.add_argument('--gen-md5', default=False, type=lambda x: (str(x).lower() == 'true'), help='')
  parser.add_argument('--dst-uri', help='GCS location from which load data')
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

  exporter = HiveExporter(project, src_dataset, src_table, tmp_dataset, tmp_table)

  if sample_size == -1 :
    if gen_md5 : exporter.select_all_with_md5(column)
    else : exporter.select_all(column)
  else :
    if gen_md5 : exporter.shuffle_sampling_with_md5(column, sample_size)
    else : exporter.shuffle_sampling(column, sample_size)

  exporter.export(dst_uri)
