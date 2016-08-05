import tensorflow as tf
import config
import model
import input
import eval

FLAGS = tf.app.flags.FLAGS

def train():
    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, FLAGS.height * FLAGS.width * FLAGS.depth], name='x-input')
        y = tf.placeholder("float", [None, FLAGS.num_class], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    hypothesis, cross_entropy, train_step = model.make_network(x, y, keep_prob)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)

        if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
            saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        for step in range(FLAGS.max_steps):
            x_data, y_data = input.get_data(sess, 'train', FLAGS.batch_size)
            if (x_data is None or y_data is None): continue
            summary, _ = sess.run([merged, train_step], feed_dict={x: x_data, y: y_data, keep_prob: 0.7})
            train_writer.add_summary(summary, step)
            print step, sess.run(cross_entropy, feed_dict={x: x_data, y: y_data, keep_prob: 1.0})

            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.checkpoint_dir + '/model.ckpt')

def main(argv = None):
    if tf.gfile.Exists(FLAGS.summary_dir + '/train'):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/train')
    tf.gfile.MakeDirs(FLAGS.summary_dir + '/train')

    if tf.gfile.Exists(FLAGS.checkpoint_dir) == False:
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()

if __name__ == '__main__':
    tf.app.run()
