class VariationalAutoencoder(object) :
    def __init__(self, sequence_length, encoding_size) :
        self.code_size = 2
        print('latent_size',self.code_size)
        self.sequence_length = sequence_length
        self.encoding_size = encoding_size
        self.data_shape = [sequence_length, encoding_size]
        self.is_training = True

        kernel_height = max(2, int(self.sequence_length/2))
        kernel_width = max(2, int(self.sequence_length/2))
        self.kernel = (kernel_height, kernel_width)
        print('kernel',self.kernel)

        stride_vertical = 1
        stride_horizontal = 2
        self.stride = (stride_vertical, stride_horizontal)
        print('stride',self.stride)

        self.conv1_filter = 16
        self.conv2_filter = 32
        self.conv_result_height = int(sequence_length / stride_vertical / stride_vertical)
        self.conv_result_width = int(encoding_size / stride_horizontal / stride_horizontal)
        self.final_conv_shape = [self.conv_result_height, self.conv_result_width, self.conv2_filter]
        print('final_conv',self.final_conv_shape)

        self.make_encoder = tf.make_template('encoder', self.make_encoder)
        self.make_decoder = tf.make_template('decoder', self.make_decoder)

    def make_prior(self):
        code_size = self.code_size
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tfd.MultivariateNormalDiag(loc, scale)

    def make_encoder(self, data):
        code_size = self.code_size
        sequence_length = self.sequence_length
        encoding_size = self.encoding_size
        conv1_filter = self.conv1_filter
        conv2_filter = self.conv2_filter
        kernel = self.kernel
        stride = self.stride

        # conv
        x = tf.reshape(data, shape=[-1, sequence_length, encoding_size, 1])

        conv1 = tf.layers.conv2d(x, conv1_filter, kernel, stride, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),bias_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.layers.conv2d(conv1, conv2_filter, kernel, stride, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),bias_initializer=tf.contrib.layers.xavier_initializer_conv2d())

        print('input',x)
        print('output',conv1)
        print('output',conv2)

        # Flatten the data to a 1-D vector for the fully connected layer
        x = tf.contrib.layers.flatten(conv2)
        x = tf.layers.dense(x, encoding_size, tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(x, code_size)
        loc = tf.layers.dense(x, code_size)
        scale = tf.layers.dense(out, code_size, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)

    def make_decoder(self, code):
        data_shape = self.data_shape
        conv1_filter = self.conv1_filter
        conv2_filter = self.conv2_filter
        encoding_size = self.encoding_size
        final_conv_shape = self.final_conv_shape
        kernel = self.kernel
        stride = self.stride

        # deconv
        x = code
        x = tf.layers.dense(x, encoding_size, tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.dense(x, np.prod(final_conv_shape))
        x = tf.reshape(x, shape=[-1] + final_conv_shape)

        conv2 = tf.layers.conv2d_transpose(x, conv1_filter, kernel, stride, padding='same')
        conv1 = tf.layers.conv2d_transpose(conv2, 1, kernel, stride, padding='same')

        print('d_input',x)
        print('d_output',conv2)
        print('d_output',conv1)

        logit = tf.reshape(conv1, [-1] + data_shape)
        print('d_output',logit)
        return tfd.Independent(tfd.Bernoulli(logit), 2) 