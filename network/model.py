import tensorflow as tf
from utils.general import clip_0_1, tensor_to_image, imshow
import IPython.display as display

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """ input_tensor : (B, H, W, C) B: Batch_size, H: Height, W: Width, C: Channels

    """
    result = tf.einsum('bwhi, bwhj->bij', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class NeuralModel(tf.keras.models.Model):
    def __init__(self, style_image, content_image, style_layers=None, content_layers=None):
        super(NeuralModel, self).__init__()
        if style_layers is None:
            style_layers = ['block{}_conv1'.format(i) for i in range(1, 6)]
        if content_layers is None:
            content_layers = ['block5_conv2']
        self.style_image = style_image
        self.content_image = content_image
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.len_style_layers = len(style_layers)
        self.vgg.trainable = False
        self.total_variation_weight = 30
        self.style_target = self(style_image)['style']
        self.content_target = self(content_image)['content']

    def call(self, inputs):
        """inputs: float number that 0 <= inputs <= 1"""
        inputs = 255. * inputs
        input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(input)
        style_outputs, content_outputs = (outputs[:self.len_style_layers], outputs[self.len_style_layers:])
        style_outputs = [gram_matrix(output) for output in style_outputs]

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def style_content_loss(self, outputs, style_weight=1e-2, content_weight=1e4):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_target[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / self.len_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_target[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / self.len_style_layers
        loss = style_loss + content_loss
        return loss

    def run(self, epochs=10, steps_per_epoch=100, show_time=True, show=True):
        import time
        start = time.time()

        image = tf.Variable(self.content_image)
        opt = tf.keras.optimizers.Adam(.02, beta_1=.99, epsilon=1e-1)
        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = self(image)
                loss = self.style_content_loss(outputs)
                loss += self.total_variation_weight * tf.image.total_variation(image)
            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print('.', end='')
            display.clear_output(wait=True)
            msg = 'Train step: {}'.format(step)
            if show:
                imshow(tensor_to_image(image), title=msg, axis='off')
            print(msg)
        end = time.time()
        if show_time:
            print('Total time {:.2f}'.format(end - start))

