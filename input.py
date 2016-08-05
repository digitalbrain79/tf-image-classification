import os
import tensorflow as tf
import numpy as np
import config
import matplotlib.pyplot as plt
import random
import struct
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
batch_index = 0
filenames = []
mnist = None

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

def get_data_jpeg(sess, data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_set)
    max = len(filenames)

    begin = batch_index
    end = batch_index + batch_size

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([])
    y_data = np.zeros((batch_size, FLAGS.num_class))
    index = 0

    for i in range(begin, end):
        with tf.gfile.FastGFile(FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0], 'rb') as f:
            image_data = f.read()

        decode_image = tf.image.decode_jpeg(image_data, channels=FLAGS.depth)
        resized_image = tf.image.resize_images(decode_image, FLAGS.height, FLAGS.width, method=1)
        image = sess.run(resized_image)
        x_data = np.append(x_data, np.asarray(image.data, dtype='float32')) / 255
        y_data[index][filenames[i][1]] = 1
        index += 1

        # print image.shape, len(image.data)
        # im = np.reshape(image.data, (256, 256, 3))
        # plt.imshow(im)
        # plt.show()

    batch_index += batch_size

    try:
        x_data = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)
    except:
        return None, None

    return x_data, y_data

def get_data_mnist(data_set, batch_size):
    global mnist
    if (mnist is None):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    if (data_set == 'train'):
        return mnist.train.next_batch(batch_size)
    else:
        return mnist.test.images[0:5000], mnist.test.labels[0:5000]

def get_data_cifar10(sess, data_set, batch_size):
    global batch_index
    x_data = []
    y_data = []
    image_size = 32 * 32 * 3
    total_size = (1 + image_size) * batch_size

    for i in range(1, 6):
        with open('./cifar10_data/cifar-10-batches-bin/data_batch_' + str(i) + '.bin', 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
            if batch_index >= len(data):
                batch_index = 0

            while batch_index < total_size:
                label = np.zeros([10])
                label[data[batch_index]] = 1
                y_data.append(label)
                batch_index += 1
                x_data.append(np.asarray(data[batch_index:batch_index + image_size], dtype='float32') / 255)
                batch_index += image_size

    return x_data, y_data

def get_data(sess, data_set, batch_size):
    #return get_data_jpeg(sess, data_set, batch_size)
    #return get_data_mnist(data_set, batch_size)
    return get_data_cifar10(sess, data_set, batch_size)

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
    for i in range(0, 1):
        get_data(sess, 'train', FLAGS.batch_size)

if __name__ == '__main__':
    tf.app.run()
