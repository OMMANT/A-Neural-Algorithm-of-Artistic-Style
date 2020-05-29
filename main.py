import tensorflow as tf
from tensorflow.keras.applications.vgg19 import *
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math



if __name__ == '__main__':
    def white_noise():
        noise = np.random.normal(0, 1, (224, 224, 3))
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = noise * 255.
        noise = noise.astype(np.uint8)
        return noise

    def gram_matrix(input):
        channels = int(input.shape[-1])
        a = np.reshape(input, (-1, channels))
        n = np.shape(a)[0]
        gram = np.matmul(a.T, a)
        return gram / n

    def get_outputs(image):
        image_batch = tf.expand_dims(image, axis=0)
        output = model(preprocess_input(image_batch * 255.))
        outputs = [gram_matrix(out) for out in output]
        return outputs

    def get_loss(G, A):
        E = [tf.reduce_mean((o - s)**2) for o, s in zip(G, A)]
        return tf.reduce_sum(E)

    def clip_0_1(image):
        clip = tf.clip_by_value(image, clip_value_min=.0, clip_value_max=1.)
        return clip

    img_dir = './res/img/'
    img_list = [img_dir + filename for filename in os.listdir(img_dir)]
    for idx, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        img_list[idx] = img
    vgg19 = VGG19(include_top=False)
    reconstructions = ['block{}_conv1'.format(n) for n in range(1, 6)]
    reconstructions = [vgg19.get_layer(name=name).output for name in reconstructions]
    vgg19.trainable = False
    model = tf.keras.Model([vgg19.input], reconstructions)
    test_img = img_list[1]
    x = np.expand_dims(test_img, axis=0)
    x = model.predict(x)
    plt.imshow(test_img)
    plt.axis('off')
    plt.show()
    gram_mats = []
    for feature_map in x:
        feature_map = np.squeeze(feature_map, axis=0)
        filters = feature_map.shape[-1]
        gram_mats.append(gram_matrix(feature_map))
        n = math.ceil(math.sqrt(filters))

    def gram_mat(input_tensor):
        return tf.einsum('ik, jk->ij', input_tensor, input_tensor)

    for l in x:
        print(l.shape)
        gram = gram_mat(l[0])
        print(gram.shape)
        print(gram)



