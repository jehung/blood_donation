import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train_cleaned.csv'
test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
data = pd.read_csv(file_in)
test = pd.read_csv(test_file)

batch_size = 1
num_steps = 30001
numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation',
                       'If']




def make_feature(data):
    temp = (data['Months since First Donation']-data['Months since Last Donation'])/data['Number of Donations']
    temp -= data['Months since Last Donation']
    data['if'] = abs(temp)

    temp1 = data['If']
    return data



def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset.loc[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



data = make_feature(data)
y = data['Made Donation in March 2007'].astype('float32')
y = (np.arange(2) == y[:,None]).astype(np.float32)
X = data.loc[:, numerical_features].astype('float32')
test_X = np.asarray(make_feature(test).astype('float32'))


train_dataset, train_labels = randomize(X, y)
scaler = StandardScaler()
X_x = scaler.fit_transform(X)



valid_size = 200

valid_dataset = np.asarray(train_dataset)[:valid_size,:]
valid_labels = np.asarray(train_labels)[:valid_size,:]
train_dataset = np.asarray(train_dataset)[valid_size:,:]
train_labels = np.asarray(train_labels)[valid_size:,:]
print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, len(numerical_features)))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 2))
    tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)

    # new hidden layer
    hidden_nodes = 1024
    hidden_weights = tf.Variable(tf.truncated_normal([len(numerical_features), hidden_nodes]))
    hidden_biases = tf.Variable(tf.zeros([hidden_nodes]))
    hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([hidden_nodes, 2]))
    biases = tf.Variable(tf.zeros([2]))

    # Training computation.
    logits = tf.matmul(hidden_layer, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_relu = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_relu, weights) + biases)

    #test_relu = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
    #test_prediction = tf.nn.softmax(tf.matmul(test_relu, weights) + biases)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        #batch_data_rs = np.reshape(batch_data, (2, -1))
        #batch_label_rs = np.reshape(batch_labels, (2, -1))
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))