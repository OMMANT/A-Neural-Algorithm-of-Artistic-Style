from network.model import NeuralModel
import matplotlib as mpl
from utils.general import *

mpl.rcParams['figure.figsize'] = (24, 12)
mpl.rcParams['axes.grid'] = False

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = './res/img/Starry-Night-canvas-Vincent-van-Gogh-New-1889.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

model = NeuralModel(style_image, content_image)
image = model.run()