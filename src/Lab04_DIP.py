# Huỳnh Lê Minh Thịnh
# ITITIU15014
from PIL import Image
from typing import Union, Iterable
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


def to_image(data, mode=None):
    return Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype='uint8'), mode=mode)


def load_image(filename, mode='RGB'):
    return np.asarray(Image.open(filename).convert(mode), dtype='uint8')


def ex01(src1: str, src2: str, to_url: str, video_length=2.0, fps=60):
    img1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    img2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    weight = tf.placeholder(dtype=tf.float32)

    tensor = (1 - weight) * img1 + weight * img2
    frames = []  # List contains Image objects

    with tf.Session() as session:
        images = video_length * fps  # Number of images splitted.
        step = 1 / (images - 1)
        for i in np.arange(0, 1 + step, step):  # Looping for @images time.
            output = session.run(tensor, feed_dict={img1: load_image(src1),
                                                    img2: load_image(src2), weight: i})
            frames.append(to_image(output))

    height, width, layers = np.array(frames[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(to_url, fourcc, float(fps), (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    return video.release()


def ex02a(seed: Union[None, int, float, np.ndarray, Iterable]):
    rs = np.random.RandomState(seed)  # Deterministic random data
    stack = np.hstack((rs.normal(size=1000), rs.normal(loc=5, scale=2, size=1000)))
    plt.hist(stack, bins='auto')  # Arguments are passed to np.histogram
    plt.title("2A. Histogram with 'auto' bins")
    return plt.show()


def ex02b(src: str, title=None, colors=2, bits=8):  # Default 2 main colors with 8-bits color system.
    crs = colors ** bits  # Color Representation Systems
    img = cv2.imread(src, 0)
    hist, bins = np.histogram(img.flatten(), crs, [0, crs])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), crs, [0, crs], color='r')
    plt.xlim([0, crs])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.title("2B. Histogram with Cumulative distribution function (cdf)" if title is None else title)
    return plt.show(), plt.imshow(Image.open(src)), plt.show()


def ex02c(src: str, to_url: str):
    original = cv2.imread(src, 0)
    equalized = cv2.equalizeHist(original)
    cv2.imwrite(to_url, equalized)
    return ex02b(src, "2C. Original Histogram"), ex02b(to_url, "2C. Equalized Histogram")


def main():
    print("----------EXERCISE 1----------")
    ex01(src1='../img/obama.jpg', src2='../img/putin.jpg', to_url='../video/ObamaToPutin.avi', video_length=5)
    print("----------EXERCISE 2----------")
    ex02a(seed=10)
    ex02b(src='../img/lena.png')
    ex02c(src='../img/img.jpg', to_url='../img/img_equalized.png')


if __name__ == '__main__':
    main()
