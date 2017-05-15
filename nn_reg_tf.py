import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train_cleaned.csv'
test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
data = pd.read_csv(file_in)
test = pd.read_csv(test_file)

batch_size = 1
num_steps = 30001
regul_param = 0.003
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



valid_size = 200
test_size = 100

valid_dataset = np.asarray(train_dataset)[:valid_size,:]
valid_labels = np.asarray(train_labels)[:valid_size,:]
train_dataset = np.asarray(train_dataset)[valid_size:-1*test_size,:]
train_labels = np.asarray(train_labels)[valid_size:-1*test_size,:]
test_dataset = np.asarray(train_dataset)[-1*test_size:,:]
test_labels = np.asarray(train_labels)[-1*test_size:,:]
print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)
print('Test', test_dataset.shape, test_labels.shape)


batch_size = 50
#regul_param = 0.003
hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, len(numerical_features))) # ONLY DIFF FOR SGD
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 2)) # ONLY DIFF FOR SGD
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([len(numerical_features), hidden_nodes]))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes]))
    biases_2 = tf.Variable(tf.zeros([hidden_nodes]))
    weights_3 = tf.Variable(tf.truncated_normal([hidden_nodes, 2]))
    biases_3 = tf.Variable(tf.zeros([2]))

    def forward_prop_dropOut(inp):
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inp, weights_1) + biases_1), 0.5)
        h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, weights_2) + biases_2), 0.5)
        return tf.matmul(h2,weights_3) + biases_3

    def forward_prop(inp):
        h1 = tf.nn.relu(tf.matmul(inp, weights_1) + biases_1)
        return tf.matmul(h1,weights_2) + biases_2

    # Training computation.
    logits = forward_prop_dropOut(tf_train_dataset)
    # L2 regularization
    regul_term = regul_param*(tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+( tf.nn.l2_loss(weights_3)))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + regul_term
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    #train_prediction = tf.nn.softmax(logits)
    train_prediction = tf.nn.softmax(forward_prop(tf_train_dataset))
    valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset))
    test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset))


num_steps = 3001
data = np.ndarray(shape=(1+num_steps//100,4), dtype=np.float32)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size)]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            batch_score = accuracy(predictions, batch_labels)
            valid_score = accuracy(valid_prediction.eval(), valid_labels)
            test_score = accuracy(test_prediction.eval(), test_labels)
            data[step//100,:] = [step*batch_size, batch_score/100, valid_score/100, test_score/100]
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % batch_score)
            print("Validation accuracy: %.1f%%" % valid_score)
    print("Test accuracy: %.1f%%" % test_score)


fig, ax = plt.subplots()
ax.plot(data[:,0], data[:,1:4])
ax.set_title('Scores given training size')
ax.legend(('batch','valid', 'test'), loc='lower right')
ax.set_xticks(data[:,0])
ax.set_xlabel('Training size')
ax.set_ylim(0,1.01)
ax.grid()
plt.show()




# Network Parameters
n_hidden_1 = 3  # 1st layer number of features
n_hidden_2 = 3  # 2nd layer number of features
n_input = 4  # MNIST data input (img shape: 28*28)
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with ReLU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with ReLU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight &amp; bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

