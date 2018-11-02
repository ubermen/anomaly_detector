# data preprocessing
sequence_length = 20
encoding_size = 128

preprocess = Preprocessor(sequence_length, encoding_size)
train_data = preprocess.extract_from_bigquery('train', 'bi-service-155107', 'bigpi_test', 'log_anomaly_globalsignin_devicemodel_20181004_real', 100000)
test_data = preprocess.extract_from_bigquery('test', 'bi-service-155107', 'bigpi_test', 'log_anomaly_globalsignin_devicemodel_20181004_test_shuffled')

# Define the model
vae = VariationalAutoencoder(sequence_length, encoding_size)
data = tf.placeholder(tf.float32, [None, sequence_length, encoding_size])
prior = vae.make_prior()
posterior = vae.make_encoder(data)
code = posterior.sample()

# Define the loss.
decoder = vae.make_decoder(code)
likelihood = decoder.log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-elbo)
anomaly_score = -likelihood

# checkpoint
checkpoint_dir = './training_checkpoints'
saver = tf.train.Saver()

test_util = TestUtil(anomaly_score)

# initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, checkpoint_dir)

# train
total_epochs = 1
batch_size = 1000
data_count = train_data.shape[0]
slice_count = int(data_count / batch_size)
with tf.Session() as sess:
    is_training = True
    abnormal_samples = None
    saver.restore(sess, checkpoint_dir)
    for epoch in range(total_epochs):
        s = (epoch*100)%len(test_data)
        e = s + 100
        test_util.test('e_{:04d}'.format(epoch), sess, test_data[s:e])

        for i in range(slice_count):
            start_index = int(i*batch_size)
            end_index = int((i+1)*batch_size-1)
            train_batch_data = train_data[start_index:end_index]
            feed = {data: train_batch_data}
            sess.run(optimize, feed)
        save_path = saver.save(sess, checkpoint_dir)

# test
with tf.Session() as sess:
    saver.restore(sess, checkpoint_dir)
    test_util.test('test', sess, test_data[:1000])

# generation
gen_count = 10000
gen_codes = vae.make_decoder(prior.sample(gen_count)).mean()

with tf.Session() as sess:
    saver.restore(sess, checkpoint_dir)
    samples = sess.run(gen_codes)
    test_util.test('generate', sess, samples)