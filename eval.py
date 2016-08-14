import tensorflow as tf
import numpy as np
import config
import model
import datetime
import input

FLAGS = tf.app.flags.FLAGS

def eval():
    keep_prob = tf.placeholder(tf.float32)
    images, labels = input.get_data('eval', FLAGS.batch_size)
    hypothesis, cross_entropy, train_step = model.make_network(images, labels, keep_prob)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
            saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        else:
            print 'Cannot find checkpoint file: ' + FLAGS.checkpoint_dir + '/model.ckpt'
            return

        delta = datetime.timedelta()
        max_steps = 10
        true_count = 0.
        total_sample_count = max_steps * FLAGS.batch_size

        top_k_op = tf.nn.in_top_k(hypothesis, labels, 1)
        tf.train.start_queue_runners(sess=sess)

        for i in range(0, max_steps):
            start = datetime.datetime.now()
            predictions = sess.run(top_k_op, feed_dict={keep_prob: 1.0})
            true_count += np.sum(predictions)
            delta += datetime.datetime.now() - start

    print 'total sample count: %d' % total_sample_count
    print 'precision @ 1: %f' % (true_count / total_sample_count)
    print 'evaluation time: %f seconds' % ((delta.seconds + delta.microseconds / 1E6) / max_steps)

def main(argv = None):
    eval()

if __name__ == '__main__':
    tf.app.run()
