# Huỳnh Lê Minh Thịnh
# ITITIU15014
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import random as rd
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def ex01(src: str):
    # Read the sonar dataset
    df = pd.read_csv(src)
    print(len(df.columns))
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    # Declaring function for applying one_hot_encoder
    def one_hot_encoding(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encoded = np.zeros((n_labels, n_unique_labels))
        one_hot_encoded[np.arange(n_labels), labels] = 1
        return one_hot_encoded

    # To encode the dependent variable as it has two categorical values
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encoding(y)

    # To divide the data in training and test subset
    # To use train_test_split() function from the sklearn library for dividing the dataset
    X, Y = shuffle(X, Y, random_state=1)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=42)

    # To define and initialize the variables to work with the tensors
    learning_rate = 0.1
    training_epochs = 1000

    # Array to store cost obtained in each epoch
    cost_history = np.empty(shape=[1], dtype=float)
    n_dim = X.shape[1]
    n_class = 2
    x = tf.placeholder(tf.float32, [None, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # To initialize all variables.
    init = tf.global_variables_initializer()

    y_ = tf.placeholder(tf.float32, [None, n_class])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)), reduction_indices=[1]))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # To initialize the session
    sess = tf.Session()
    sess.run(init)
    mse_history = []

    # To calculate the cost for each epoch
    for epoch in range(training_epochs):
        sess.run(training_step, feed_dict={x: train_x, y_: train_y})
        cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
        cost_history = np.append(cost_history, cost)
        print('epoch : ', epoch, ' - ', 'cost: ', cost)

    # Run the trained model on test subset
    predict_y = sess.run(y, feed_dict={x: test_x})
    # To calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(predict_y, 1),
                                  tf.argmax(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy))

    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    return plt.show()


def ex2a(arr: np.ndarray):
    print("*Input >")
    pprint(arr)
    print("*Output >")
    firstMap = list(map(lambda x: np.bincount(x, minlength=arr.max() + 1).tolist(), arr))
    lastMap = list(map(lambda x: x[arr.min():], firstMap))  # Remove first n-elements of each array if count == 0 !
    pprint(lastMap)
    return lastMap


def ex2b(arr: np.ndarray):
    def isValid(index, value, array):
        try:
            return array[index - 1] < value and value > arr[index + 1]
        except IndexError:
            return False

    peeks: list
    try:
        peeks = [i for i, v in enumerate(arr) if isValid(i, v, arr)]
    except IndexError:
        print("ARRAY INDEX OUT OF BOUND ERROR !")
    print("Peek Index:", peeks)
    print("Peek Value:", list(map(lambda x: arr[x], peeks)))
    return peeks


def ex02a(arr: np.ndarray):
    output = [[list(i).count(j) for j in range(1, 10)] for i in arr]
    cell = tf.Variable(output)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(cell))


def ex02b(arr: np.ndarray):
    peaks = arr[1:-1][np.diff(np.diff(arr)) < 0]
    peak_values = tf.Variable(np.nonzero(np.in1d(arr, peaks))[0], name='peak_values')
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(peak_values))


def main():
    rd.seed(100)
    print("----------EXERCISE 1----------")
    ex01(src="../data/sonar.all-data")
    print("----------EXERCISE 2A----------")
    ex2a(rd.randint(1, 11, size=(6, 10)))
    ex02a(rd.randint(1, 11, size=(6, 10)))
    print("----------EXERCISE 2B----------")
    ex2b(np.array([1, 3, 7, 1, 2, 6, 0, 1]))
    ex02b(np.array([1, 3, 7, 1, 2, 6, 0, 1]))


if __name__ == '__main__':
    main()
