# Huỳnh Lê Minh Thịnh
# ITITIU15014
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def to_image(data, mode=None):
    return Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype='uint8'), mode=mode)


def load_image(filename, mode='RGB'):
    return np.asarray(Image.open(filename).convert(mode), dtype='uint8')


def ex01(src: str):
    def to_grey_scale(filename):
        r, g, b = Image.open(filename).split()
        return 0.299 * np.asarray(r) + 0.587 * np.asarray(g) + 0.114 * np.asarray(b)

    grey_img = tf.placeholder(dtype=tf.float32, shape=[None, None])
    d = tf.placeholder(dtype=tf.float32)

    tensor = grey_img * d / 255

    with tf.Session() as session:
        output = session.run(tensor, feed_dict={grey_img: to_grey_scale(src), d: 255})
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(src))  # Before modifying
        plt.subplot(1, 2, 2)
        plt.imshow(Image.fromarray(output))  # After modifying
        return plt.show()


def ex02(src: str):
    img = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)

    tensor = img * a + b

    with tf.Session() as session:
        output = session.run(tensor, feed_dict={img: load_image(src), a: 2.2, b: 50})
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(src))  # Before modifying
        plt.subplot(1, 2, 2)
        plt.imshow(to_image(output))  # After modifying
        return plt.show()


def ex03(src1: str, src2: str, cross_dissolve: float):
    img1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    img2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    weight = tf.placeholder(dtype=tf.float32)

    tensor = (1 - weight) * img1 + weight * img2

    with tf.Session() as session:
        output = session.run(tensor, feed_dict={img1: load_image(src1),
                                                img2: load_image(src2),
                                                weight: cross_dissolve})
        plt.imshow(to_image(output))
        plt.show()


def main():
    print("----------EXERCISE 1----------")
    ex01(src='../img/HappyFish.jpg')
    print("----------EXERCISE 2----------")
    ex02(src='../img/lena.png')
    print("----------EXERCISE 3----------")
    ex03(src1='../img/LinuxLogo.jpg', src2='../img/WindowsLogo.jpg', cross_dissolve=0.5)


if __name__ == '__main__':
    main()
