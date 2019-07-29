# Huỳnh Lê Minh Thịnh
# ITITIU15014
import numpy as np
import tensorflow as tf


def ex01():
    # Creating variable for parameter slope (W) with initial value as 0.4
    W = tf.Variable([0.4], dtype=tf.float32)
    # Creating variable for parameter bias (b) with initial value as -0.4
    b = tf.Variable([-0.4], dtype=tf.float32)
    # Creating placeholders for providing input or independent variable, denoted by x
    x = tf.placeholder(tf.float32)
    # Equation of Linear Regression
    linear_model = W * x + b
    # Initializing all the variables
    initialized_variables = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(initialized_variables)
        # Running regression model to calculate the output w.r.t. to provided x values
        print(session.run(linear_model, feed_dict={x: [1, 2, 3, 4, 5, 6, 7]}))


def ex02():
    train_x = np.linspace(-1, 1, 101)
    train_y = 3 * train_x + np.random.randn(*train_x.shape) * 0.33

    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weights")

    y_model = tf.multiply(X, w)
    cost = (tf.pow(Y - y_model, 2))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            for (x, y) in zip(train_x, train_y):
                sess.run(train_op, feed_dict={X: x, Y: y})
                print(sess.run(w))


def ex03():
    a = tf.constant(0, name='a')
    b = tf.constant(1, name='b')
    c = tf.constant(2, name='c')
    d = tf.constant(3, name='d')
    e = tf.constant(4, name='e')
    f = tf.constant(5, name='f')
    g = tf.constant(6, name='g')

    graph = tf.constant([[0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0]], dtype=tf.int32)

    def edge(v1, v2):
        return session.run(graph[v1][v2])

    def has_path(v1, v2):
        if edge(v1, v2):
            return True
        else:
            for i in [a, b, c, d, e, f, g]:
                if v1 != i and edge(v1, i) and has_path(i, v2):
                    return True
            return False

    def all_paths(v1, v2, path, paths):
        path = path + [v1.name]
        if v1 == v2:
            paths.append(path)
        else:
            for i in [a, b, c, d, e, f, g]:
                if edge(v1, i):
                    all_paths(i, v2, path, paths)
        return paths

    def cycles():
        for i in [a, b, c, d, e, f, g]:
            paths = all_paths(i, i, path=[], paths=[])
            if (paths != []) is True:
                print(paths)

    with tf.Session() as session:
        print(all_paths(v1=a, v2=e, path=[], paths=[]))
        print(session.run(all_paths(v1=a, v2=e, path=[], paths=[])))


def main():
    print("----------EXERCISE 1----------")
    ex01()
    print("----------EXERCISE 2----------")
    ex02()
    print("----------EXERCISE 3----------")
    ex03()


if __name__ == '__main__':
    main()
