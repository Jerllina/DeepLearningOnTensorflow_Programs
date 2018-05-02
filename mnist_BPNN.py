import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
#28*28
OUTPUT_NODE = 10
#0~9
LAYER1_NODE = 500
BATCH_SIZE = 10

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    '''input layer'''
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    '''bp 1'''
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    tf.summary.histogram("weights1", weights1)
    tf.summary.histogram("biases1", biases1)
    
    '''bp 2'''
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    tf.summary.histogram("weights2", weights2)
    tf.summary.histogram("biases2", biases2)

    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regulatization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regulatization
    #loss = cross_entropy_mean
    tf.summary.scalar('loss', loss)

    #learning_rate = LEARNING_RATE_BASE
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/tmp/xx", sess.graph)
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)

                print("After %d training step(s), validation accuracy"
                      "using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary,_=sess.run([merged,train_op], feed_dict={x: xs, y_: ys})
            writer.add_summary(summary, i)

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print ("After %d traning step(s), test accuracy using average"
               "model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    start = time.clock()
    mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
    train(mnist)
    end = time.clock()
    print('finish all in %s' % str(end - start))

if __name__ == '__main__':

    tf.app.run()
