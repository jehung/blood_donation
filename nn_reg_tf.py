import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


file_in = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\train_cleaned.csv'
test_file = 'C:\\Users\\Jenny\\Documents\\Mathfreak_Data\\DataKind\\BloodDonation\\test.csv'
data = pd.read_csv(file_in)
test = pd.read_csv(test_file)

num_labels = 2
num_steps = 5001
beta_regul = 1e-3
valid_size = 200
numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation',
                       'If']




def make_feature(data):
    temp = (data['Months since First Donation']-data['Months since Last Donation'])/data['Number of Donations']
    temp -= data['Months since Last Donation']
    data['if'] = abs(temp)

    temp1 = data['If']
    return data


def transform(data):
    return PolynomialFeatures(2).fit_transform(data)


def reformat_train(dataset, labels):
    dataset = dataset.reshape((-1, len(dataset[0]))).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


def reformat_test(dataset):
    dataset = dataset.reshape((-1, len(dataset[0]))).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    return dataset



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



data = make_feature(data)
y = data['Made Donation in March 2007'].astype('float32')
X = data.loc[:, numerical_features].astype('float32')
X = transform(X)
test_X = make_feature(test)
test1 = test_X.loc[:, numerical_features].astype('float32')
test1 = transform(test1)


train_dataset, train_labels = reformat_train(X, y)
test_dataset = reformat_test(test1)



valid_dataset = np.asarray(train_dataset)[:valid_size,:].astype(np.float32)
valid_labels = np.asarray(train_labels)[:valid_size,:].astype(np.float32)
train_dataset = np.asarray(train_dataset)[valid_size:,:].astype(np.float32)
train_labels = np.asarray(train_labels)[valid_size:,:].astype(np.float32)
final_test = np.asarray(test1).astype(np.float32)
print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)
print('Final Test', final_test.shape, final_test.shape)



graph = tf.Graph()
batch_size = 20
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 256
num_hidden_nodes3 = 128
keep_prob = 0.5

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=[None, train_dataset.shape[1]])
    tf_train_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.placeholder(tf.float32,
                                      shape=[None, test_dataset.shape[1]])
    global_step = tf.Variable(0)

    # Variables.
    weights1 = tf.Variable(
        tf.truncated_normal(
            [train_dataset.shape[1], num_hidden_nodes1],
            stddev=np.sqrt(2.0 / train_dataset.shape[1]))
    )
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
    weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
    weights3 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
    weights4 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)))
    biases4 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
    lay3_train = tf.nn.relu(tf.matmul(lay2_train, weights3) + biases3)
    logits = tf.matmul(lay3_train, weights4) + biases4
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.0003, global_step, 1000, 0.90, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
    lay3_valid = tf.nn.relu(tf.matmul(lay2_valid, weights3) + biases3)
    valid_prediction = tf.nn.softmax(tf.matmul(lay3_valid, weights4) + biases4)
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
    lay3_test = tf.nn.relu(tf.matmul(lay2_test, weights3) + biases3)
    test_prediction = tf.nn.softmax(tf.matmul(lay3_test, weights4) + biases4)



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
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


    ## Submission
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
    lay3_test = tf.nn.relu(tf.matmul(lay2_test, weights3) + biases3)
    logits = tf.matmul(lay3_test, weights4) + biases4
    test_prediction = tf.nn.softmax(logits)
    print(test_prediction)

    a = session.run(test_prediction, feed_dict={tf_test_dataset: test_dataset})
    print(a)
'''    
    for i in range(train_dataset.shape[0]):
        #for start, end in zip(range(0, train_dataset.shape[0], batch_size),range(batch_size, train_dataset.shape[0] + 1, batch_size)):
        a = session.run(test_prediction, feed_dict={tf_test_dataset: test_dataset})
#       print(i, np.mean(np.argmax(teY, axis=1) ==
#                         sess.run(predict_op, feed_dict={X: teX})))
        print(a)


    feed_dict = {tf_train_dataset: test_dataset}
    #proba = session.run(final_test_prediction)
    proba = test_prediction.eval(feed_dict)
    print('proba', proba)
    classification = session.run(tf.argmax(test_prediction, 1), feed_dict=feed_dict)
    print('class', classification)


    test_col = pd.DataFrame(proba)
    df_id = test.loc[:, ['ID']]
    test_mid = pd.concat([df_id, test_col], axis=1)
    test_mid.head()
    submission = test_mid.loc[:, ['ID', 1]]
    submission.rename(columns={'ID': '', 1: 'Made Donation in March 2007'}, inplace=True)
    submission.to_csv('submission.csv')



fig, ax = plt.subplots()
ax.plot(data[:,0], data[:,1:4])
ax.set_title('Scores given training size')
ax.legend(('batch','valid', 'test'), loc='lower right')
ax.set_xticks(data[:,0])
ax.set_xlabel('Training size')
ax.set_ylim(0,1.01)
ax.grid()
plt.show()


'''