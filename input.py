import os
import tensorflow as tf
import numpy as np
import random
import config

FLAGS = tf.app.flags.FLAGS

def get_filenames(data_set):
    classes = []
    filenames = []
    labels = []

    with open(FLAGS.data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            classes += inner_list

    for i, lable in enumerate(classes):
        list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + lable)
        for filename in list:
            filenames.append(lable + '/' + filename)
            labels.append(i)

    return filenames, labels

def read_jpeg(data_set, batch_size):
    filenames, labels = get_filenames(data_set)
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=FLAGS.depth)

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        sess.run(image)

    return image, labels[0]

def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(batch_size):
    image, label = read_jpeg('train', batch_size)
    reshaped_image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(reshaped_image, FLAGS.height + 32, FLAGS.width + 32)

    distorted_image = tf.random_crop(resized_image, [FLAGS.height, FLAGS.width, FLAGS.depth])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    return generate_image_and_label_batch(float_image, label, 1000, batch_size, shuffle=True)

def inputs(batch_size):
    image, label = read_jpeg('eval', batch_size)
    reshaped_image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.height, FLAGS.width)

    float_image = tf.image.per_image_whitening(resized_image)

    return generate_image_and_label_batch(float_image, label, 100, batch_size, shuffle=False)

def get_data(data_set, batch_size):
    if data_set is 'train':
        return distorted_inputs(batch_size)
    else:
        return input(batch_size)

def main(argv = None):
    get_data('train', 10)

if __name__ == '__main__':
    tf.app.run()
