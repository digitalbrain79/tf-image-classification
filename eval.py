import tensorflow as tf
import config
import model
import datetime
import input

FLAGS = tf.app.flags.FLAGS

def eval():
    x_data, y_data = input.get_data()

    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, FLAGS.height * FLAGS.width * FLAGS.depth], name='x-input')
        y = tf.placeholder("float", [None, FLAGS.num_class], name='y-input')

    hypothesis, cross_entropy, train_step = model.make_network(x, y)

    sess = tf.Session()
    saver = tf.train.Saver()

    if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
        saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
    else:
        print 'Cannot find checkpoint file: ' + FLAGS.checkpoint_dir + '/model.ckpt'
        return

    test_data = x_data
    for i in range(FLAGS.num_samples):
        for j in range(FLAGS.height * FLAGS.width * FLAGS.depth):
            if (j % 5 == 0):
                test_data[i][j] = 0

    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    start = datetime.datetime.now()
    result = sess.run(accuracy, feed_dict={x: test_data, y: y_data})
    delta = datetime.datetime.now() - start

    print 'accuracy: %f' % result
    print 'evaluation time: %f seconds' % (delta.seconds + delta.microseconds / 1E6)

def main(argv = None):
    if tf.gfile.Exists(FLAGS.eval_summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_summary_dir)
    tf.gfile.MakeDirs(FLAGS.eval_summary_dir)
    eval()

if __name__ == '__main__':
    tf.app.run()
