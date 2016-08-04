import os
import tensorflow as tf
import numpy as np
import config
import matplotlib.pyplot as plt
import random

FLAGS = tf.app.flags.FLAGS
batch_index = 0
filenames = []

def get_filenames(data_set):
    global filenames
    labels = []

    with open(FLAGS.data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list

    for i, lable in enumerate(labels):
        list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + lable)
        for filename in list:
            filenames.append([lable + '/' + filename, i])

    random.shuffle(filenames)

def get_data(sess, data_set):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_set)
    max = len(filenames)

    begin = batch_index
    end = batch_index + FLAGS.batch_size

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([])
    y_data = np.zeros((FLAGS.batch_size, FLAGS.num_class))
    index = 0

    for i in range(begin, end):
        image_data = tf.gfile.FastGFile(FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0], 'r').read()
        decode_image = tf.image.decode_jpeg(image_data, channels=FLAGS.depth)
        resized_image = tf.image.resize_images(decode_image, FLAGS.height, FLAGS.width, method=1)
        image = sess.run(resized_image)
        x_data = np.append(x_data, np.asarray(image.data, dtype='float32')) / 255
        y_data[index][filenames[i][1]] = 1
        index += 1

        #print image.shape, len(image.data)
        #im = np.reshape(image.data, (256, 256))
        #plt.imshow(im)
        #plt.show()

    x_data = x_data.reshape(FLAGS.batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)
    batch_index += FLAGS.batch_size

    return x_data, y_data

"""
def get_data(sess):
    x_data = np.array([])
    y_data = np.zeros((FLAGS.num_samples, FLAGS.num_class))

    array_size = FLAGS.height * FLAGS.width * FLAGS.depth

    for i in range(0, FLAGS.num_samples):
        data = np.random.rand(array_size).astype("float32")
        x_data = np.append(x_data, data)
        y_data[i][i % FLAGS.num_class] = 1

    x_data = x_data.reshape(FLAGS.num_samples, array_size)

    return x_data, y_data
"""

def main(argv = None):
    sess = tf.Session()
    for i in range(0, 10):
        get_data(sess, 'train')

if __name__ == '__main__':
    tf.app.run()
