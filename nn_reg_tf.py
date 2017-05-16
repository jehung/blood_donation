import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train_cleaned.csv'
test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
data = pd.read_csv(file_in)
test = pd.read_csv(test_file)

batch_size = 50
num_steps = 30001
regul_param = 0
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
print('y', y)
X = data.loc[:, numerical_features].astype('float32')

test_X = make_feature(test)
test = test_X.loc[:, numerical_features].astype('float32')


train_dataset, train_labels = randomize(X, y)



valid_size = 200


valid_dataset = np.asarray(train_dataset)[:valid_size,:]
valid_labels = np.asarray(train_labels)[:valid_size,:]
train_dataset = np.asarray(train_dataset)[valid_size:,:]
train_labels = np.asarray(train_labels)[valid_size:,:]
all_train = np.asarray(train_dataset)
all_train_labels = np.asarray(train_labels)
final_test = np.asarray(test)
print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)
print('Final Test', final_test.shape, final_test.shape)


batch_size = 200
hidden_nodes = 5

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.00001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               50, 0.96, staircase=True)
    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, len(numerical_features))) # ONLY DIFF FOR SGD
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 2)) # ONLY DIFF FOR SGD
    tf_valid_dataset = tf.constant(valid_dataset)
    all_train = tf.constant(all_train)
    all_train_labels = tf.constant(all_train_labels)
    tf_test_dataset = tf.constant(final_test)

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
        h2 = tf.nn.relu(tf.matmul(h1, weights_2) + biases_2)
        return tf.matmul(h2,weights_3) + biases_3

    def lof_loss(labels, softmax):
        return -1*(labels * softmax).sum()

    # Training computation.
    logits = forward_prop_dropOut(tf_train_dataset)
    # L2 regularization
    regul_term = regul_param*(tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+( tf.nn.l2_loss(weights_3)))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + regul_term
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Predictions for the training, validation, and test data.
    #train_prediction = tf.nn.softmax(logits)
    train_prediction = tf.nn.softmax(forward_prop_dropOut(tf_train_dataset))
    valid_prediction = tf.nn.softmax(forward_prop_dropOut(tf_valid_dataset))
    final_test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset))
    log_loss = -1*tf.reduce_mean(tf_train_labels*train_prediction)

    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss))


data = np.ndarray(shape=(1+num_steps//100,3), dtype=np.float32)

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
        if (step % 50 == 0):
            batch_score = accuracy(predictions, batch_labels)
            batch_loss = tf.contrib.losses.log_loss(tf_train_labels, train_prediction)
            valid_score = accuracy(valid_prediction.eval(), valid_labels)
            #valid_loss = tf.contrib.losses.log_loss(valid_labels, valid_prediction)
            data[step//100,:] = [step*batch_size, batch_score/100, valid_score/100]
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % batch_score)
            print("Validation accuracy: %.1f%%" % valid_score)



    ## Submission
    final_test_prediction = tf.nn.softmax(forward_prop(final_test))
    feed_dict = {tf_train_dataset: final_test}
    proba = final_test_prediction.eval(feed_dict)
    print('proba', proba)
    classification = session.run(tf.argmax(final_test_prediction, 1), feed_dict=feed_dict)
    print('class', classification)





fig, ax = plt.subplots()
ax.plot(data[:,0], data[:,1:4])
ax.set_title('Scores given training size')
ax.legend(('batch','valid', 'test'), loc='lower right')
ax.set_xticks(data[:,0])
ax.set_xlabel('Training size')
ax.set_ylim(0,1.01)
ax.grid()
plt.show()


