import PIL.Image
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.config.run_functions_eagerly(True)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")  # todo: del this
        loss = tf.constant(0.0)
        for n in tqdm(tf.range(steps)):
            with tf.GradientTape() as tape:
                # we use gradient tape to watch gradients relative to input image
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)

            # normalize gradients
            # earlier we removed mean from losses. does it mean that gradients have already mean = 0?
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # perform gradient ascent
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


class TiledDeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # init the image gradients to zero
        gradients = tf.zeros_like(img_rolled)

        # get tiles coords
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not len(xs):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not len(ys):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)

                    img_tile = img_rolled[y:y + tile_size, x:x + tile_size]
                    loss = calc_loss(img_tile, self.model)

                gradients = gradients + tape.gradient(loss, img_rolled)

        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        return gradients


def random_roll(img, maxroll):
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled


def run_deep_dream_simple(img, deep_dream_obj, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)

        steps_remaining -= run_steps
        step += run_steps

        loss, img = deep_dream_obj(img, run_steps, tf.constant(step_size))

    result = deprocess(img)

    return result


def run_octave_deep_dream(img, deep_dream_obj, octave_scale, octave_powers, steps_per_octave=50, step_size=0.01):
    img = tf.constant(np.array(img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in octave_powers:
        new_shape = tf.cast(float_base_shape * (octave_scale ** n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(
            img,
            deep_dream_obj,
            steps_per_octave,
            step_size
        )

    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)

    return img


def run_random_octave_deep_dream(
        img, gradient_obj, steps_per_octave=100, step_size=1e-2, octaves=range(-2, 3), octave_scale=1.3,
        model_type="inception",
        specified_channel=None
):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    if model_type in ["inceptionv3", "efficientnet", "efficientnetv2b2"]:
        img = tf.keras.applications.inception_v3.preprocess_input(img)
    elif model_type in ["resnet50v2"]:
        img = tf.keras.applications.resnet_v2.preprocess_input(img)

    # clip model cond
    clip_0_255 = ["resnext"]
    clip_model_cond = any(list(map(lambda x: x in model_type.lower(), clip_0_255)))

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        print(f"Octave {octave}")
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)

        for step in tqdm(range(steps_per_octave)):
            gradients = gradient_obj(img, new_size)
            img = img + gradients * step_size
            if clip_model_cond:
                img = tf.clip_by_value(img, 0, 255)
            else:
                img = tf.clip_by_value(img, -1, 1)

    result = deprocess(img)
    return result


def decode_vgg16(img):
    if isinstance(img, np.ndarray):
        out = np.copy(img)
    else:
        out = np.copy(img.numpy())
    means_bgr = [103.939, 116.779, 123.68]
    out[:, :, 0] += means_bgr[0]
    out[:, :, 1] += means_bgr[1]
    out[:, :, 2] += means_bgr[2]
    # bgr -> rgb
    out = out[..., ::-1]
    # clip
    out = tf.clip_by_value(out, 0, 255).numpy()
    return out


def different_deep_dream(
        img, gradient_obj, steps_per_octave=10, step_size=1e-2, octave_n=4,
        octave_scale=1.4, model_type="inceptionv3", specified_channel=None
):
    img = tf.keras.utils.img_to_array(img)
    if model_type in ["inceptionv3", "efficientnet", "efficientnetv2b2"]:
        img = tf.keras.applications.inception_v3.preprocess_input(img)
    elif model_type in ["resnet50v2"]:
        img = tf.keras.applications.resnet_v2.preprocess_input(img)
    elif model_type in ["vgg16"]:
        img = tf.keras.applications.vgg16.preprocess_input(img)

    # clip model cond
    clip_0_255 = ["resnext"]
    clip_model_cond = any(list(map(lambda x: x in model_type.lower(), clip_0_255)))

    img0 = img

    octaves = []
    for i in range(octave_n - 1):
        hw = img0.shape[:2]
        new_size = tf.cast(tf.convert_to_tensor(hw), tf.float32) / octave_scale
        new_size = tf.cast(new_size, tf.int32)
        lo = tf.image.resize(img0, new_size)
        hi = img0 - tf.image.resize(lo, hw)
        img0 = lo
        octaves.append(hi)

    for octave in range(octave_n):
        print(f"Octave {octave}")
        if octave > 0:
            hi = octaves[-octave]
            img0 = tf.image.resize(img0, hi.shape[:2]) + hi
        for i in tqdm(range(steps_per_octave)):
            # calc the grad here
            grad = gradient_obj(img0, img0.shape[:2])
            # img0 += grad * (step_size / np.abs(grad).mean() + 1e-7)
            img0 += grad * step_size

            if model_type in ["vgg16"]:
                means_bgr = [103.939, 116.779, 123.68]
                means = np.ones(img0.shape)
                means[:, :, 0] *= means_bgr[0]
                means[:, :, 1] *= means_bgr[1]
                means[:, :, 2] *= means_bgr[2]
                img0 += means
                img0 = tf.clip_by_value(img0, 0, 255)
                img0 -= means
            elif clip_model_cond:
                img0 = tf.clip_by_value(img0, 0, 255)
            else:
                img0 = tf.clip_by_value(img0, -1, 1)

    if model_type in ["vgg16"]:
        result = decode_vgg16(img0).astype(np.uint8)
    else:
        result = deprocess(img0)
    return result


# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def open_image(path, max_dim=None):
    img = PIL.Image.open(path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# Display an image
def show(img):
    plt.imshow(img)
    plt.show()


def calc_loss(img, model, specified_channel=None):
    img_batch = tf.expand_dims(img, axis=0)
    if specified_channel is not None:
        layer_activations = model(img_batch)[:, :, :, specified_channel]
    else:
        layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    # normalize losses here to stop larger layers outweighting smaller layers
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDreamer:
    def __init__(self):
        self.models = {
            "inceptionv3": {
                "model": tf.keras.applications.InceptionV3(
                    include_top=False,
                    weights="imagenet",
                ),
                # "model": None,
                "layers": ["mixed0", "mixed3", "mixed5"],
            },  # best step_size = 1e-2, steps=50
            "efficientnetv2b2": {
                "model": tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
                    include_top=False,
                    weights='imagenet',
                    include_preprocessing=False
                ),
                # "model": None,
                "layers": ["block4d_add", "block6b_add"]
            },
            "vgg16": {
                "model": tf.keras.applications.vgg16.VGG16(
                    include_top=False,
                    weights="imagenet",
                ),
                # "model": None,
                "layers": ["block3_conv3", "block4_conv1", "block4_conv3", "block5_conv3"]
            }
            # best vgg params:
            # step_size = 1.0
            # steps = 15
            # rest is the same as for the rest
        }

    def perform_deep_dream(
            self, raw_img, picked_model, layer_names, steps_per_octave, step_size=1., num_of_octaves=4, octave_scale=1.4
    ):
        base_model = self.models[picked_model]["model"]
        layers = [base_model.get_layer(name).output for name in layer_names]
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        gradient_model = TiledDeepDream(dream_model)

        octaved_dreamed_image = different_deep_dream(
            raw_img, gradient_model, steps_per_octave=steps_per_octave, step_size=step_size, octave_n=num_of_octaves,
            octave_scale=octave_scale, model_type=picked_model, specified_channel=None
        )

        if not isinstance(octaved_dreamed_image, np.ndarray):
            octaved_dreamed_image = octaved_dreamed_image.numpy()
        return octaved_dreamed_image

# default parameters
# step_size: 1.
# num_of_octaves: 4
# octave_scale: 1.4

def main():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    # original_img = download(url, max_dim=500)
    path = "/Users/gwilczynski/Desktop/xd2.png"
    original_img = open_image(path, max_dim=None)
    # show(original_img)
    model_type = "vgg16"
    num_of_steps = 15
    step_size = 1.
    if model_type == "inceptionv3":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            # input_shape=original_img.shape
        )
    elif model_type == "efficientnetv2b2":
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
            include_top=False,
            weights='imagenet',
            include_preprocessing=False
        )
    elif model_type == "resnet50v2":
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
            # input_shape=original_img.shape,
            include_top=False,
            weights="imagenet"
        )
    elif model_type == "convnext_base":
        base_model = tf.keras.applications.convnext.ConvNeXtBase(
            model_name='convnext_base',
            include_top=False,
            weights='imagenet',
        )
    elif model_type == "vgg16":
        base_model = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights="imagenet",
            # input_shape=original_img.shape,
        )

    # layer_names = ["mixed3", "mixed5"]
    # major_layer_names = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    # major_layer_names = ["block4_conv1", "block4_conv2", "block5_conv1", "block5_conv2"]
    major_layer_names = ["block5_conv1"]
    specified_channel = None
    # major_layer_names = ["conv5_block1_2_conv", "conv5_block2_2_conv", "conv4_block1_3_conv", "conv3_block3_out", "conv3_block3_1_conv"]

    for layer_names in major_layer_names:
        print(f"{layer_names}")
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        layers = [base_model.get_layer(name).output for name in layer_names]

        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        deep_dream_model = DeepDream(dream_model)
        gradient_model = TiledDeepDream(dream_model)

        # dreamed_image = run_deep_dream_simple(
        #     original_img,
        #     deep_dream_model,
        #     steps=100,
        #     step_size=0.01
        # )
        # octaved_dreamed_image = run_octave_deep_dream(
        #     original_img,
        #     deep_dream_model,
        #     1.3,
        #     [-2, -1, 0, 1, 2],
        #     50,
        #     1e-2
        # )
        # octaved_dreamed_image = run_random_octave_deep_dream(
        #     original_img,
        #     gradient_model,
        #     100,
        #     1e-2,
        #     range(-2, 3),
        #     1.3
        # )
        # octaved_dreamed_image = run_random_octave_deep_dream(
        #     original_img,
        #     gradient_model,
        #     num_of_steps,
        #     1e-2,
        #     range(-2, 3),
        #     1.3,
        #     model_type,
        #     specified_channel
        # )
        octaved_dreamed_image = different_deep_dream(
            original_img, gradient_model, steps_per_octave=num_of_steps, step_size=step_size, octave_n=4,
            octave_scale=1.4, model_type=model_type, specified_channel=None
        )
        # show(octaved_dreamed_image)
        # save
        if isinstance(octaved_dreamed_image, np.ndarray):
            pil_img = PIL.Image.fromarray(octaved_dreamed_image)
        else:
            pil_img = PIL.Image.fromarray(octaved_dreamed_image.numpy())
        if specified_channel is not None:
            pil_img.save(
                f"dreamed-{model_type}-{layer_names[0]}-steps{num_of_steps}-chan{specified_channel}-step_size{step_size}.png")
        else:
            pil_img.save(f"new-dreamed-{model_type}-{layer_names[0]}-steps{num_of_steps}-step_size{step_size}.png")


if __name__ == '__main__':
    main()
