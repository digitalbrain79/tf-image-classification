import tensorflow as tf
import numpy as np

height = 48
width = 48
depth = 3
num_class = 10
num_samples = 100
array_size = height * width * depth

x_data = np.array([])
y_data = np.zeros((num_samples, num_class))

for i in range(0, num_samples):
  data = np.random.rand(array_size).astype("float32")
  x_data = np.append(x_data, data)
  y_data[i][i % num_class] = 1

x_data = x_data.reshape(num_samples, array_size)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", [None, array_size])
y = tf.placeholder("float", [None, num_class])
x_image = tf.reshape(x, [-1, height, width, depth])

W_conv1 = weight_variable([3, 3, depth, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

height /= 4
width /= 4

W_fc1 = weight_variable([height * width * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, height * width * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, num_class])
b_fc2 = bias_variable([num_class])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1000):
  sess.run(train_step, feed_dict={x: x_data, y: y_data})
  print step, sess.run(cross_entropy, feed_dict={x: x_data, y: y_data})

test_data = x_data
for i in range(num_samples):
  for j in range(array_size):
    if (j % 5 == 0):
      test_data[i][j] = 0

prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: test_data, y: y_data})