import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from concurrent.futures import ThreadPoolExecutor
import utils


class MobileNetV2Model(object):

    def __init__(self):
        encoder_nn = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        self.encoder = tf.keras.Model(encoder_nn.input,
                                      encoder_nn.layers[-2].output)
        self.embedding_dim = self.encoder.output.shape[1]

    def load_img(self, fname, target_img_size=(224, 224)):
        img = load_img(fname, target_size=target_img_size)
        img = img_to_array(img)
        return img

    def embed(
        self,
        filenames,
        batch_size=64,
        l2_norm=False,
        n_preprocess_workers=8
    ):
        with ThreadPoolExecutor(max_workers=n_preprocess_workers) as executor:
            images = [i for i in executor.map(self.load_img, filenames)]

        if len(images) > batch_size:
            batches = utils.chunks(images, batch_size)
        else:
            batches = [images]
        vectors = []
        for batch in batches:
            batch = preprocess_input(np.array(batch))
            batch = self.encoder.predict(batch, batch_size=batch_size)
            vectors.append(batch)

        if l2_norm:
            vectors = [utils.l2norm(v) for v in vectors]

        return vectors
