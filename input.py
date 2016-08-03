import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def get_data():
    x_data = np.array([])
    y_data = np.zeros((FLAGS.num_samples, FLAGS.num_class))

    array_size = FLAGS.height * FLAGS.width * FLAGS.depth

    for i in range(0, FLAGS.num_samples):
        data = np.random.rand(array_size).astype("float32")
        x_data = np.append(x_data, data)
        y_data[i][i % FLAGS.num_class] = 1

    x_data = x_data.reshape(FLAGS.num_samples, array_size)

    return x_data, y_data
