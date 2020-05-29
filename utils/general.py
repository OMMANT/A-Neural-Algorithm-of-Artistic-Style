import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    tensor = tensor * 255.
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None, gray=False, axis='on'):
    assert axis in ['on', 'off']
    plt.axis(axis)
    if gray:
        image = tf.image.rgb_to_grayscale(image)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
