import tensorflow as tf
import config
import model
import input
import eval

FLAGS = tf.app.flags.FLAGS

def train():
    keep_prob = tf.placeholder(tf.float32)
    images, labels = input.get_data('train', FLAGS.batch_size)
    hypothesis, cross_entropy, train_step = model.make_network(images, labels, keep_prob)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
            saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            sess.run(train_step, feed_dict={keep_prob: 0.7})
            print step, sess.run(cross_entropy, feed_dict={keep_prob: 1.0})

            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.checkpoint_dir + '/model.ckpt')

def main(argv = None):
    if tf.gfile.Exists(FLAGS.checkpoint_dir) == False:
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()

if __name__ == '__main__':
    tf.app.run()
