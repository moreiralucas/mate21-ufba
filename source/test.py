# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load all images from the test set (folder of images).                                              #
# Parameters:                                                                                                #
#         path - path to the folder                                                                          #
#         height - number of image rows                                                                      #
#         width - number of image columns                                                                    #
#         num_channels - number of image channels                                                            #
# Return values:                                                                                             #
#         X - ndarray with all images                                                                        #
# ---------------------------------------------------------------------------------------------------------- #
def load_test_set(path, height, width, num_channels):
    images = sorted(os.listdir(path))
    num_images = len(images)

    X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)

    for i in range(num_images):
        img = cv2.imread(path+'/'+images[i], cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
        assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[i]
        assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[i]
        X[i] = img

    return X, images

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Netork parameters                                                                                  #
# ---------------------------------------------------------------------------------------------------------- #
IMAGE_HEIGHT = 77         # height of the image
IMAGE_WIDTH = 71          # width of the image
NUM_CHANNELS = 1          # number of channels of the image
NUM_CLASSES = 10          # number of classes

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load the test set                                                                                  #
# ---------------------------------------------------------------------------------------------------------- #
X_img, X_name = load_test_set(sys.argv[1], IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
X_img = X_img/255.

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Inference graph                                                                                    #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    y = tf.placeholder(tf.int64, shape = (None,))
    y_one_hot = tf.one_hot(y, NUM_CLASSES)
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    X = tf.layers.dropout(X, 0.2, training=is_training) # Dropout
    out = tf.layers.conv2d(X, 32, (3, 3), (1, 1), padding='valid', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='valid')           
    out = tf.layers.conv2d(out, 64, (3, 3), (2, 2), padding='valid', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='valid')
    out = tf.layers.conv2d(out, 128, (3, 3), (2, 2), padding='valid', activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (3, 3), (2, 2), padding='valid')
    out = tf.layers.dropout(out, 0.3, training=is_training) # Dropout
    out = tf.reshape(out, [-1, out.shape[1]*out.shape[2]*out.shape[3]])
    out = tf.layers.dense(out, NUM_CLASSES, activation=None)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_one_hot, out))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    result = tf.argmax(out, 1)
    correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Test loop                                                                                          #
# ---------------------------------------------------------------------------------------------------------- #
with tf.Session(graph = graph) as session:
    # model saver
    saver = tf.train.Saver(max_to_keep=0)
    saver.restore(session, './model/cnn')
    fp = open(sys.argv[2],'w')
    for i in range(len(X_name)):
        ret = session.run([result], feed_dict = {X: X_img[i:i+1], is_training: False})
        fp.write(X_name[i]+' '+str(ret[0][0])+'\n')
    fp.close()

