#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import sys
import random
from clusterone import get_data_path

def r_square(X, Y):
    avx = np.mean(X)
    avy = np.mean(Y)
    sum1, sumx, sumy = 0, 0, 0
    for i in range(len(X)):
        sum1 += (X[i] - avx)*(Y[i] - avy)
        sumx += (X[i] - avx)*(X[i] - avx)
        sumy += (Y[i] - avy)*(Y[i] - avy)
    return sum1 * sum1 / (sumx * sumy)


sys.stdout.write("reading data ... ")
sys.stdout.flush()
start = datetime.datetime.now()

# file path when running on clusterone: /data/my_username/dataset_name/
data_path = get_data_path(
            dataset_name = 'zhaoxiaq/qianqianmerckdata',  # on ClusterOne
            local_root = '~/',  # path to local dataset
            local_repo = 'TrainingSet',  # local data folder name
            path = 'ACT1_competition_training.csv'  # folder within the data folder
            )

train_1 = pd.read_csv(data_path, engine='python', dtype={"MOLECULE": object, "Act": float})
# file path when running on my own pc or cloud
# train_1 = pd.read_csv('TrainingSet/ACT1_competition_training.csv',dtype={"MOLECULE": object, "Act": float})

stop = datetime.datetime.now()
sys.stdout.write("done\n")
sys.stdout.write("took {} seconds\n".format((stop - start).total_seconds()))
sys.stdout.flush()

y = train_1['Act'].values
y = np.reshape(y, (-1, 1))
train_1 = train_1.drop(['Act', 'MOLECULE'], axis = 1)
train_1 = train_1.apply(lambda x: np.log(x+1))
x = train_1.values

# randomize train/test split in each run
seed = round(random.uniform(1, len(x)))
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.80, random_state = seed)

# define placeholders for input x and y
X_placeholder = tf.placeholder(tf.float64, (None, X_train.shape[1]))
Y_placeholder = tf.placeholder(tf.float64, (None, Y_train.shape[1]))

# define parameters
features = np.shape(X_train)[1]
target_size = np.shape(X_train)[0]

learning_rate = 0.001
epochs = 600
batch_size = 300

batch_size_placeholder = tf.placeholder(tf.int64)

# network parameters
n_hidden_1 = 50
n_hidden_2 = 25

ds_train = tf.data.Dataset.from_tensor_slices((X_placeholder, Y_placeholder)).shuffle(buffer_size=round(len(X_train) * 0.3)).batch(batch_size_placeholder)

ds_test = tf.data.Dataset.from_tensor_slices((X_placeholder, Y_placeholder)).batch(batch_size_placeholder)

ds_iter = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)

next_x, next_y = ds_iter.get_next()

train_init_op = ds_iter.make_initializer(ds_train)
test_init_op = ds_iter.make_initializer(ds_test)

# define placeholder for input vector X and target vector y
keep_prob = tf.placeholder(tf.float64)

# initialize weights and bias  
weights = {'w1': tf.Variable(tf.truncated_normal([features, n_hidden_1], 0, 1, dtype=tf.float64)),
           'w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], 0, 1, dtype=tf.float64)),
          'out': tf.Variable(tf.truncated_normal([n_hidden_2, 1], 0, 1, dtype=tf.float64))}

biases = {'b1': tf.Variable(tf.truncated_normal([n_hidden_1], 0, 1, dtype=tf.float64)),
          'b2': tf.Variable(tf.truncated_normal([n_hidden_2], 0, 1, dtype=tf.float64)),
         'out': tf.Variable(tf.truncated_normal([1], 0, 1, dtype=tf.float64))}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer 1 with ReLu activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob) 
    
    # Hidden layer 2 with ReLu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob) 
    
    # Output layer with ReLu activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer
              
# construct model
y_pred = multilayer_perceptron(next_x, weights, biases)

# define cost function(mean squred error) and optimizer(gradient descent)
cost =  tf.losses.mean_squared_error(next_y, y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(epochs):
        # random shuffle data in each epoch
        sess.run(train_init_op, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train, batch_size_placeholder: batch_size})
        count = 0

        while True:
            try:
                count += 1
                _, c = sess.run((optimizer, cost), feed_dict={keep_prob: 0.75})
                print('Epoch:', (epoch + 1), 'Batch:', count, 'cost =', c)
            except tf.errors.OutOfRangeError:
                break

    sess.run(test_init_op, feed_dict={X_placeholder: X_test, Y_placeholder: Y_test, batch_size_placeholder: len(X_test)})

    results, test_cost = sess.run((y_pred, cost), feed_dict={keep_prob: 1.0})

    #print(results)
    print(test_cost)
    print('R^2:', r_square(np.reshape(results, (len(results),)), Y_test))

