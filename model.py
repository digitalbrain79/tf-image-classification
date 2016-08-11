import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, height, width):
    return tf.nn.max_pool(x, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='SAME')

def make_network(images, labels, keep_prob):
    num_conv = FLAGS.num_conv
    kernel_size = FLAGS.kernel_size
    pool_size = FLAGS.pool_size
    num_map = FLAGS.num_map
    num_fc_layer = FLAGS.num_fc_layer
    num_fc_input = FLAGS.num_fc_input

    height = FLAGS.height
    width = FLAGS.width
    prev_num_map = FLAGS.depth
    h_pool = images

    for i in range(num_conv):
        W_conv = weight_variable([kernel_size, kernel_size, prev_num_map, num_map], 'W_conv' + str(i+1))
        b_conv = bias_variable([num_map], 'b_conv' + str(i+1))
        h_conv = tf.nn.relu(conv2d(h_pool, W_conv) + b_conv)
        h_pool = max_pool(h_conv, pool_size, pool_size)
        prev_num_map = num_map
        num_map *= 2
        height /= 2
        width /= 2

    num_map /= 2
    h_fc_input = tf.reshape(h_pool, [-1, height * width * num_map])
    prev_num_fc_input = height * width * num_map

    for i in range(num_fc_layer):
        W_fc = weight_variable([prev_num_fc_input, num_fc_input], 'W_fc' + str(i+1))
        b_fc = bias_variable([num_fc_input], 'b_fc' + str(i+1))
        h_fc = tf.nn.relu(tf.matmul(h_fc_input, W_fc) + b_fc)
        h_fc_input = tf.nn.dropout(h_fc, keep_prob)
        prev_num_fc_input = num_fc_input
        num_fc_input /= 2

    num_fc_input *= 2
    W_fc = weight_variable([num_fc_input, FLAGS.num_class], 'W_fc' + str(i+2))
    b_fc = bias_variable([FLAGS.num_class], 'b_fc' + str(i+2))

    hypothesis = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.matmul(h_fc_input, W_fc) + b_fc, labels)
    cross_entropy = tf.reduce_mean(hypothesis)
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    return cross_entropy, train_step
