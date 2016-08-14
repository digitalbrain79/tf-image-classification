import tensorflow as tf
import config

FLAGS = tf.app.flags.FLAGS

def read_raw_images(data_set):
    filename = ['./data/' + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)

    image_bytes = FLAGS.raw_height * FLAGS.raw_width * FLAGS.depth
    record_bytes = image_bytes + 1
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    depth_major = tf.reshape(tf.slice(record_bytes, [1], [image_bytes]),
        [FLAGS.depth, FLAGS.raw_height, FLAGS.raw_width])
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    return uint8image, label

def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(batch_size):
    image, label = read_raw_images('train')
    reshaped_image = tf.cast(image, tf.float32)

    distorted_image = tf.random_crop(reshaped_image, [FLAGS.height, FLAGS.width, FLAGS.depth])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    return generate_image_and_label_batch(float_image, label, 100, batch_size, shuffle=True)

def inputs(batch_size):
    image, label = read_raw_images('eval')
    reshaped_image = tf.cast(image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.height, FLAGS.width)
    float_image = tf.image.per_image_whitening(resized_image)

    return generate_image_and_label_batch(float_image, label, 10, batch_size, shuffle=False)

def get_data(data_set, batch_size):
    if data_set is 'train':
        return distorted_inputs(batch_size)
    else:
        return inputs(batch_size)

def main(argv = None):
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        for i in range(0, 10):
            result = get_data('train', 10)
            print result

if __name__ == '__main__':
    tf.app.run()
