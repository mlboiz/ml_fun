import functools
import os

import numpy as np
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()


def main():
    content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'
    style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
    path_for_hub_models = ""
    os.environ["TFHUB_CACHE_DIR"] = path_for_hub_models

    out_image_size = 384

    content_img_size = (out_image_size, out_image_size)
    style_img_size = (256, 256)

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding="SAME")

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]

    show_n([content_image, style_image, stylized_image], ["content image", "style image", "stylized image"])


class StyleTransfer:
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

    def __init__(self, path_for_hub_models, out_image_size=384, style_img_size=256):
        os.environ["TFHUB_CACHE_DIR"] = path_for_hub_models
        self.content_img_size = (out_image_size, out_image_size)
        self.style_img_size = (style_img_size, style_img_size)
        self.hub_module = hub.load(StyleTransfer.hub_handle)

    def stylize(self, content_image, style_image):
        content_image = self.preprocess_image(content_image, self.content_img_size)
        style_image = self.preprocess_image(style_image, self.style_img_size)
        outputs = self.hub_module(content_image, style_image)
        return (np.array(outputs[0])[0]*256).astype(np.uint8)

    @staticmethod
    def preprocess_image(raw_image, image_size=(256, 256), preserve_aspect_ratio=True):
        img = tf.convert_to_tensor(raw_image, dtype=tf.float32)[tf.newaxis, ...]
        img = crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
        img = tf.cast(img, dtype=tf.float32) / tf.constant(256, dtype=tf.float32)
        return img


if __name__ == '__main__':
    main()
