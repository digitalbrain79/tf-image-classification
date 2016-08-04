import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('height', 256, '')
tf.app.flags.DEFINE_integer('width', 256, '')
tf.app.flags.DEFINE_integer('depth', 1, '')
tf.app.flags.DEFINE_integer('num_class', 3, '')
tf.app.flags.DEFINE_integer('num_samples', 100, '')
tf.app.flags.DEFINE_integer('num_map1', 8, '')
tf.app.flags.DEFINE_integer('num_map2', 16, '')
tf.app.flags.DEFINE_integer('num_fc', 256, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_integer('batch_size', 10, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')

tf.app.flags.DEFINE_string('train_summary_dir', './summary/train', '')
tf.app.flags.DEFINE_string('eval_summary_dir', './summary/eval', '')
tf.app.flags.DEFINE_string('data_dir', './data', '')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', '')