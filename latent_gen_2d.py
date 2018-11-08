import tensorflow as tf

class LatentGen2D (object) :
  def __init__(self, size=10, interval=0.1, horizontal_offset=0, vertical_offset=0, print_result=True) :
    half_size = int(size/2)
    width = 2*half_size + 1
    height = width
    count = width*height

    v1_offset = horizontal_offset
    v2_offset = vertical_offset
    v1_start = -interval*half_size + v1_offset
    v2_start = -interval*half_size + v2_offset

    codes = [0] * count
    for i in range(height) :
      v2 = v2_start + i*interval
      for j in range(width) :
        v1 = v1_start + j*interval
        ptr = i*width+j
        codes[ptr] = [v2,v1]
    
    self.codes = codes
    self.width = width
    
  def print_codes(self) :
    self.print_row_by_row(self.codes, self.width)
    
  def print_ascii(self, sess, generator, test_util):
    manual_codes = tf.placeholder(tf.float32, [None, code_size])
    gen_codes = generator.make_decoder(manual_codes).mean()
    samples = sess.run(gen_codes, {manual_codes:self.codes})
    ascii_codes_list = test_util.convert_onehot_to_ascii(sess, samples)
    self.print_row_by_row(ascii_codes_list, self.width)
    
  def print_row_by_row(self, arr, width):
    line = ''
    count = 0
    for v in arr :
      line = line + '\t' + str(v)
      count = count + 1
      if count == width :
        print(line)
        count = 0
        line = ''
