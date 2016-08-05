import tensorflow as tf
import config
import model
import datetime
import input

FLAGS = tf.app.flags.FLAGS

def eval():
    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, FLAGS.height * FLAGS.width * FLAGS.depth], name='x-input')
        y = tf.placeholder("float", [None, FLAGS.num_class], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    hypothesis, cross_entropy, train_step = model.make_network(x, y, keep_prob)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
            saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        else:
            print 'Cannot find checkpoint file: ' + FLAGS.checkpoint_dir + '/model.ckpt'
            return

        accuracy = 0
        delta = datetime.timedelta()
        max_steps = 10

        for i in range(0, max_steps):
            x_data, y_data = input.get_data(sess, 'eval', FLAGS.batch_size)
            prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
            accuracy += tf.reduce_mean(tf.cast(prediction, tf.float32))
            start = datetime.datetime.now()
            result = sess.run(accuracy, feed_dict={x: x_data, y: y_data, keep_prob: 1.0})
            delta += datetime.datetime.now() - start

    print 'accuracy: %f' % (result / max_steps)
    print 'evaluation time: %f seconds' % ((delta.seconds + delta.microseconds / 1E6) / max_steps)

def main(argv = None):
    eval()

if __name__ == '__main__':
    tf.app.run()
