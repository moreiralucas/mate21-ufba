import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os

from data import Dataset
from parameters import Parameters


import cv2
# p = Parameters()
# d = Dataset()
# train, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
# train = d.shuffle(train[0], train[1], seed=42)

# train, val = d.split(train[0], train[1], p.SPLIT_RATE)
# train = d.augmentation(train[0], train[1])

class Net():
    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Load the training set, shuffle its images and then split them in training and validation subsets.  #
    #         After that, load the testing set.                                                                  #
    # ---------------------------------------------------------------------------------------------------------- #
    def __init__(self, p, size_class_train=10):
        # ---------------------------------------------------------------------------------------------------------- #
        # Description:                                                                                               #
        #         Create a training graph that receives a batch of images and their respective labels and run a      #
        #         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
        #         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
        # ---------------------------------------------------------------------------------------------------------- #
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape = (None, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS))
            self.y = tf.placeholder(tf.int64, shape = (None,))
            self.y_one_hot = tf.one_hot(self.y, size_class_train)
            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            print(self.X.shape)

            self.X = tf.layers.dropout(self.X, 0.2, training=self.is_training) # Dropout
            self.out = tf.layers.conv2d(self.X, 32, (3, 3), (1, 1), padding='valid', activation=tf.nn.relu)
            print(self.out.shape)
            self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='valid')
            print(self.out.shape)
            
            self.out = tf.layers.conv2d(self.out, 64, (3, 3), (2, 2), padding='valid', activation=tf.nn.relu)
            self.out = tf.layers.max_pooling2d(self.out, (2, 2), (2, 2), padding='valid')
            print(self.out.shape)
            
            self.out = tf.layers.conv2d(self.out, 128, (3, 3), (2, 2), padding='valid', activation=tf.nn.relu)
            print(self.out.shape)
            self.out = tf.layers.max_pooling2d(self.out, (3, 3), (2, 2), padding='valid')

            self.out = tf.layers.dropout(self.out, 0.3, training=self.is_training) # Dropout
            print(self.out.shape)

            self.out = tf.reshape(self.out, [-1, self.out.shape[1]*self.out.shape[2]*self.out.shape[3]])

            self.out = tf.layers.dense(self.out, size_class_train, activation=None)

            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_one_hot, self.out))

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.result = tf.argmax(self.out, 1)
            self.correct = tf.reduce_sum(tf.cast(tf.equal(self.result, self.y), tf.float32))

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Training loop.                                                                                     #
    # ---------------------------------------------------------------------------------------------------------- #
    def treino(self, input_train, input_val):
        self.train = input_train
        self.val = input_val
        p = Parameters()
        with tf.Session(graph = self.graph) as session:
            # weight initialization
            session.run(tf.global_variables_initializer())

            menor_loss = 1e9
            best_acc = 0
            epoca = 0
            saver = tf.train.Saver()

            # full optimization
            for epoch in range(p.NUM_EPOCHS_FULL):
                print('Epoch: '+ str(epoch+1), end=' ')
                lr = (p.S_LEARNING_RATE_FULL*(p.NUM_EPOCHS_FULL-epoch-1)+p.F_LEARNING_RATE_FULL*epoch)/(p.NUM_EPOCHS_FULL-1)
                self.training_epoch(session, self.train_op, lr)
                val_acc, val_loss = self.evaluation(session, self.val[0], self.val[1], name='Validation')
                # Otimizar o early stopping
                if val_acc > best_acc:
                    menor_loss = val_loss
                    best_acc = val_acc
                    epoca = epoch
                    saver.save(session, os.path.join(p.LOG_DIR, 'model/cnn'))
                    print ('The model has successful saved')
                # else:
                #     contador += 1
                #     if contador > p.TOLERANCE:
                #         print('The train has stopped')
                #         break
                #         print('O treino deveria ter parado se estivesse usando o early stopping')
                #     print ('The model hasn\'t saved')
                # break #TODO
                print ('\n-********************************************************-')

            print ("Best_acc : " + str(best_acc) + ", loss: " + str(menor_loss) + ", epoca: " + str(epoca)) 
    
    # def prediction(self, test, classes_train):
    #     print ('-********************************************************-')
    #     print ('Start prediction ...')
    #     #p = Parameters()
    #     with tf.Session(graph = self.graph) as session:
    #         outputs = None
    #         time_now = datetime.datetime.now()
    #         path_txt = str(time_now.day) + '_' + str(time_now.hour) + 'h'  + str(time_now.minute) + 'm.txt'
    #         with open(path_txt, 'w') as f:
    #             for j in range(len(test[0])):
    #                 feed_dict={self.X: np.reshape(test[0][j], (1, ) + test[0][j].shape), self.is_training: False}
    #                 saida = session.run(self.out, feed_dict)
    #                 outputs = np.array(saida[0])
    #                 resp = str(test[1][j]) +' ' + str(np.argmax(outputs)) + '\n'
    #                 f.write(resp)
    #             f.close()
    
    def prediction2(self, test):
        print("prediction2(self, test)")
        p = Parameters()

        with tf.Session(graph = self.graph) as session:
            saver = tf.train.Saver(max_to_keep=0)
            # print (os.path.join(p.LOG_DIR, 'cnn'))
            saver.restore(session, './model/cnn')
            time_now = datetime.datetime.now()
            path_txt = str(time_now.day) + 'd' + str(time_now.hour) + 'h'  + str(time_now.minute) + 'm.txt'
            # print("path_txt: {}".format(path_txt))
            # cont = 0
            predicao = []
            with open(path_txt, 'w') as f:
                for j in range(len(test[0])):
                    feed_dict={self.X: np.reshape(test[0][j], (1, ) + test[0][j].shape), self.is_training: False}
                    saida = session.run([self.result], feed_dict)
                    # print(saida[0][0])
                    # outputs = np.array(saida[0])
                    resp = str(test[1][j]) +' ' + str(saida[0][0]) + '\n'
                    predicao.append(resp)
                    # f.write(resp)
                    # print("resp: " + resp )
                    # if cont == 1:
                    #     break
                    # if cont % 100 == 0:
                    #     cv2.imshow(resp, test[0][j])
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                    # cont += 1
                predicao.sort()
                for j in range(len(predicao)):
                    f.write(predicao[j])
                f.close()
                # cv2.destroyAllWindows()
    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Evaluate images in Xv with labels in yv.                                                           #
    # ---------------------------------------------------------------------------------------------------------- #
    def evaluation(self, session, Xv, yv, name='Evaluation'):
        p = Parameters()
        start = time.time()
        eval_loss = 0
        eval_acc = 0
        for j in range(0, len(Xv), p.BATCH_SIZE):
            ret = session.run([self.loss, self.correct], feed_dict = {self.X: Xv[j:j+p.BATCH_SIZE], self.y: yv[j:j+p.BATCH_SIZE], self.is_training: False})
            eval_loss += ret[0]*min(p.BATCH_SIZE, len(Xv)-j)
            eval_acc += ret[1]

        print('Time:'+str(time.time()-start)+' ACC:'+str(eval_acc/len(Xv))+' Loss:'+str(eval_loss/len(Xv)))
        return eval_acc/len(Xv), eval_loss/len(Xv)

    # ---------------------------------------------------------------------------------------------------------- #
    # Description:                                                                                               #
    #         Run one training epoch using images in X_train and labels in y_train.                              #
    # ---------------------------------------------------------------------------------------------------------- #
    def training_epoch(self, session, op, lr):
        batch_list = np.random.permutation(len(self.train[0]))
        p = Parameters()
        start = time.time()
        train_loss = 0
        train_acc = 0
        for j in range(0, len(self.train[0]), p.BATCH_SIZE):
            if j+p.BATCH_SIZE > len(self.train[0]):
                break
            X_batch = self.train[0].take(batch_list[j:j+p.BATCH_SIZE], axis=0)
            y_batch = self.train[1].take(batch_list[j:j+p.BATCH_SIZE], axis=0)

            ret = session.run([op, self.loss, self.correct], feed_dict = {self.X: X_batch, self.y: y_batch, self.learning_rate: lr, self.is_training: True})
            train_loss += ret[1]*p.BATCH_SIZE
            train_acc += ret[2]

        pass_size = (len(self.train[0])-len(self.train[0])%p.BATCH_SIZE)
        print('LR:'+str(lr)+' Time:'+str(time.time()-start)+' ACC:'+str(train_acc/pass_size)+' Loss:'+str(train_loss/pass_size))
