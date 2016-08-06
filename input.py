import os
import tensorflow as tf
import numpy as np
import config
import matplotlib.pyplot as plt
import random
import struct
from tensorflow.examples.tutorials.mnist import input_data
import pickle

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

        try:
            decode_image = tf.image.decode_jpeg(image_data, channels=FLAGS.depth)
            resized_image = tf.image.resize_images(decode_image, FLAGS.height, FLAGS.width, method=1)
            image = sess.run(resized_image)
        except:
            print 'invalid jpeg image'
            continue

        x_data = np.append(x_data, np.asarray(image.data, dtype='float32'))# / 255
        y_data[index][filenames[i][1]] = 1
        index += 1

        #print image.shape, len(image.data)
        #im = np.reshape(image.data, (FLAGS.height, FLAGS.width, FLAGS.depth))
        #plt.imshow(im)
        #plt.show()

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
        return mnist.test.images, mnist.test.labels

cifar10_data = np.array([], dtype=np.uint8)

def load_cifar10_data(data_set):
    global cifar10_data

    if (data_set == 'train'):
        for i in range(1, 6):
            with open('./cifar10_data/cifar-10-batches-bin/data_batch_' + str(i) + '.bin', 'rb') as f:
                cifar10_data = np.append(cifar10_data, np.fromfile(f, dtype=np.uint8))
    else:
        with open('./cifar10_data/cifar-10-batches-bin/test_batch.bin', 'rb') as f:
            cifar10_data = np.fromfile(f, dtype=np.uint8)

def get_data_cifar10(sess, data_set, batch_size):
    global batch_index, cifar10_data
    x_data = []
    y_data = []
    image_size = 32 * 32 * 3
    total_size = (1 + image_size) * batch_size

    if len(cifar10_data) == 0:
        load_cifar10_data(data_set)

    if batch_index >= len(cifar10_data):
        batch_index = 0

    end = batch_index + total_size

    while batch_index < end:
        label = np.zeros([10])
        label[cifar10_data[batch_index]] = 1
        y_data.append(label)
        batch_index += 1
        x_data.append(np.asarray(cifar10_data[batch_index:batch_index + image_size], dtype=np.float32) / 255)
        batch_index += image_size

    return x_data, y_data

image_data = []
label_data = []

def get_data_raw(data_set, batch_size):
    global image_data, label_data, batch_index

    if len(label_data) == 0:
        if data_set == 'train':
            image_filename = FLAGS.data_dir + '/train_image.bin'
            label_filename = FLAGS.data_dir + '/train_label.bin'
        else:
            image_filename = FLAGS.data_dir + '/eval_image.bin'
            label_filename = FLAGS.data_dir + '/eval_label.bin'

        with open(image_filename, 'rb') as image_file, open(label_filename, 'rb') as label_file:
            image_data = pickle.load(image_file)
            label_data = pickle.load(label_file)

    if batch_index >= len(label_data):
        batch_index = 0

    x_data = image_data[batch_index:batch_index + batch_size]
    y_data = label_data[batch_index:batch_index + batch_size]
    batch_index += batch_size

    return x_data, y_data

def get_data(sess, data_set, batch_size):
    #return get_data_jpeg(sess, data_set, batch_size)
    return get_data_raw(data_set, batch_size)
    #return get_data_mnist(data_set, batch_size)
    #return get_data_cifar10(sess, data_set, batch_size)

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
    x_data = []
    y_data = []

    with tf.Session() as sess:
        with open(FLAGS.data_dir + '/eval_image.bin', 'wb') as image_file, open(FLAGS.data_dir + '/eval_label.bin', 'wb') as label_file:
            for i in range(0, 300):
                print 'step %d' % i
                image, label = get_data(sess, 'eval', 1)
                if image != None:
                    x_data.append(image[0])
                    y_data.append(label[0])

            pickle.dump(x_data, image_file)
            pickle.dump(y_data, label_file)

if __name__ == '__main__':
    tf.app.run()
