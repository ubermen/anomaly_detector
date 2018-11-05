import pandas as pd
import numpy as np

class Preprocessor(object) :
    def __init__(self, sequence_length, encoding_size) :
        self.sequence_length = sequence_length
        self.encoding_size = encoding_size

    def is_alphabet(self, ascii_code) :
        if ascii_code is not None and ((65 <= ascii_code <= 90) or (97 <= ascii_code <= 122)) : return 1
        else : return 0

    def is_number(self, ascii_code) :
        if ascii_code is not None and (48 <= ascii_code <= 57) : return 1
        else : return 0

    def convert_str_df_to_onehot_ndarray(self, tag, df) :
        raw = df.values
        return self.convert_str_ndarray_to_onehot_ndarray(tag, raw)

    def convert_str_array_to_onehot_ndarray(self, tag, arr) :
        raw = np.array(arr)
        return self.convert_str_ndarray_to_onehot_ndarray(tag, raw)

    def convert_str_ndarray_to_onehot_ndarray(self, tag, raw) :
        data_count = raw.shape[0]
        charseqs = np.zeros((data_count, self.sequence_length, self.encoding_size))
        for i, v in enumerate(raw) :
            value = str(v[0])
            if value is None : continue
            instance_length = min(self.sequence_length, len(value))
            for j in range(instance_length) :
                unicode = ord(value[j])
                if unicode >= self.encoding_size : continue
                charseqs[i][j][unicode] = 1
        return charseqs.reshape(data_count, self.sequence_length, self.encoding_size).astype('float32')

    def get_df_from_bigquery(self, project_id, dataset, table, limit=None) :
        query = 'SELECT * FROM [{}.{}.{}]'.format(project_id, dataset, table)
        if limit is not None : query = query + ' limit {}'.format(limit)
        df = pd.io.gbq.read_gbq(query, project_id=project_id, verbose=False)
        return df

    def extract_from_bigquery(self, tag, project_id, dataset, table, limit=None, print_result=False) :
        df = self.get_df_from_bigquery(project_id, dataset, table, limit)
        nd = self.convert_str_df_to_onehot_ndarray(tag, df)
        if print_result : print(tag, nd.shape[0])
        df = None
        return nd
    
    def get_df_batch_from_bigquery(self, project_id, dataset, table, min_rnum, max_rnum) :
        min_rnum += 1
        max_rnum += 1
        query = 'SELECT * FROM (SELECT *, ROW_NUMBER() OVER() as rnum FROM [{}.{}.{}]) WHERE {} <= rnum AND rnum <= {}'.format(project_id, dataset, table, min_rnum, max_rnum)
        df = pd.io.gbq.read_gbq(query, project_id=project_id, verbose=False)
        return df

    def extract_batch_from_bigquery(self, tag, project_id, dataset, table, min_rnum, max_rnum, print_result=False) :
        df = self.get_df_batch_from_bigquery(project_id, dataset, table, min_rnum, max_rnum)
        nd = self.convert_str_df_to_onehot_ndarray(tag, df)
        if print_result : print(tag, min_rnum, max_rnum)
        df = None
        return nd
