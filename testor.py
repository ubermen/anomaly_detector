class TestUtil(object) :
    def __init__(self, anomaly_score) :
        self.anomaly_score = anomaly_score

    def match_and_sort(self, key, value) :
        matched = [(i,j) for i,j in zip(key, value)]
        sorted_by_value = sorted(matched, key=lambda x: x[1])
        return sorted_by_value

    def convert_onehot_to_ascii(self, sess, onehot_array) :
        sequence_length = len(onehot_array[0])
        result = []
        max_indexes = tf.argmax(onehot_array,2)
        ascii_codes_list = sess.run(max_indexes)
        for index, ascii_codes in enumerate(ascii_codes_list) :
            v1 = ''
            for ascii_code in ascii_codes :
                if ascii_code == 0 : continue
                v1 = v1 + chr(ascii_code)
            result.append(v1)
        return result

    def get_score_list(self, sess, samples):
        feed = {data: samples}
        score = sess.run([self.anomaly_score], feed)
        return score[0]

    def test(self, tag, sess, samples) :
        ascii_codes_list = self.convert_onehot_to_ascii(sess, samples)
        score = self.get_score_list(sess, samples)
        sorted_by_score = self.match_and_sort(ascii_codes_list, score)
        print(sorted_by_score)