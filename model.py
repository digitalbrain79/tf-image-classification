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

def max_pool(x, height, width, depth):
    return tf.nn.max_pool(x, ksize=[1, height, width, depth], strides=[1, height, width, depth], padding='SAME')

def make_network(x, y):
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, FLAGS.height, FLAGS.width, FLAGS.depth])
        tf.image_summary('input', x_image, FLAGS.num_class)

    W_conv1 = weight_variable([3, 3, FLAGS.depth, FLAGS.num_map1], 'W_conv1')
    b_conv1 = bias_variable([FLAGS.num_map1], 'b_conv1')

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, 2, 2, 1)

    W_conv2 = weight_variable([3, 3, FLAGS.num_map1, FLAGS.num_map2], 'W_conv2')
    b_conv2 = bias_variable([FLAGS.num_map2], 'b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, 2, 2, 1)

    height = FLAGS.height / 4
    width = FLAGS.width / 4

    W_fc1 = weight_variable([height * width * FLAGS.num_map2, FLAGS.num_fc], 'W_fc1')
    b_fc1 = bias_variable([FLAGS.num_fc], 'b_fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, height * width * FLAGS.num_map2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([FLAGS.num_fc, FLAGS.num_class], 'W_fc2')
    b_fc2 = bias_variable([FLAGS.num_class], 'b_fc2')

    hypothesis = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))
        tf.scalar_summary('cross entropy', cross_entropy)
    train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    return hypothesis, cross_entropy, train_step
