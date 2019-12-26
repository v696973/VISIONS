# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import utils


def cosine_sim(img_v, text_v):
    return tf.matmul(img_v, text_v, transpose_b=True)


def vsepp_mh_loss(img_v, text_v, margin=0.2):
    '''
    TensorFlow 2.0 implementation of Max of Hinges contrastive loss
    described in https://arxiv.org/abs/1707.05612
    see https://github.com/fartashf/vsepp/blob/master/model.py#L245
    for Pytorch reference implementation
    '''

    sim_scores = cosine_sim(img_v, text_v)
    diagonal = tf.reshape(tf.linalg.tensor_diag_part(sim_scores),
                          (img_v.get_shape()[0], 1))
    d1 = tf.broadcast_to(diagonal, sim_scores.get_shape())
    d2 = tf.broadcast_to(tf.transpose(diagonal), sim_scores.get_shape())

    cost_text = tf.nn.relu((margin + sim_scores - d1))
    cost_img = tf.nn.relu((margin + sim_scores - d2))

    # set diagonals to zeros
    cost_img = tf.linalg.set_diag(cost_img,
                                  tf.zeros(diagonal.get_shape()[0]))
    cost_text = tf.linalg.set_diag(cost_text,
                                   tf.zeros(diagonal.get_shape()[0]))

    cost_text = tf.reduce_max(cost_text, axis=1)
    cost_img = tf.reduce_max(cost_img, axis=0)
    # return (tf.reduce_sum(cost_img) + tf.reduce_sum(cost_text)) / img_v.get_shape()[0]
    # loss value depends on a batch size
    return tf.reduce_sum(cost_img) + tf.reduce_sum(cost_text)


class VSEPPModel(object):

    def __init__(
        self,
        img_encoder,
        text_encoder,
        aligned_embedding_dim=512,
        img_aligner_weights=None,
        text_aligner_weights=None,
        normalize_img_aligner_input=True,
        normalize_img_aligner_output=True,
        normalize_text_aligner_input=True,
        normalize_text_aligner_output=True
    ):
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.aligned_embedding_dim = aligned_embedding_dim
        self.img_aligner = self.create_img_aligner(
            weights_path=img_aligner_weights,
            normalize_input=normalize_img_aligner_input,
            normalize_output=normalize_img_aligner_output
        )
        self.text_aligner = self.create_text_aligner(
            weights_path=text_aligner_weights,
            normalize_input=normalize_text_aligner_input,
            normalize_output=normalize_text_aligner_output
        )

    def create_img_aligner(
        self,
        weights_path=None,
        normalize_input=True,
        normalize_output=True
    ):
        # img aligner model takes image encoder output and projects it
        # into a common visual-semantic embedding space
        aligner_layers = []
        aligner_layers.append(
            tf.keras.layers.Input(
                shape=(self.img_encoder.embedding_dim,)
            )
        )
        if normalize_input:
            aligner_layers.append(
                tf.keras.layers.Lambda(utils.tf_l2norm)
            )
        aligner_layers.append(
            tf.keras.layers.Dense(
                self.aligned_embedding_dim,
                activation='linear'
            )
        )
        if normalize_output:
            aligner_layers.append(
                tf.keras.layers.Lambda(utils.tf_l2norm)
            )
        aligner_model = tf.keras.Sequential(aligner_layers)
        if weights_path:
            aligner_model.load_weights(weights_path)
        return aligner_model

    def create_text_aligner(
        self,
        normalize_input=True,
        normalize_output=True,
        weights_path=None
    ):
        # text aligner model takes text encoder output and projects it
        # into a common visual-semantic embedding space
        aligner_layers = []

        aligner_layers.append(
            tf.keras.layers.Input(
                shape=(self.text_encoder.embedding_dim,)
            )
        )
        if normalize_input:
            aligner_layers.append(
                tf.keras.layers.Lambda(utils.tf_l2norm)
            )
        aligner_layers.append(
            tf.keras.layers.Dense(
                self.aligned_embedding_dim,
                activation='linear'
            )
        )
        if normalize_output:
            aligner_layers.append(
                tf.keras.layers.Lambda(utils.tf_l2norm)
            )
        aligner_model = tf.keras.Sequential(aligner_layers)
        if weights_path:
            aligner_model.load_weights(weights_path)
        return aligner_model

    def save_img_aligner_weights(self, fpath):
        self.img_aligner.save_weights(fpath)

    def save_text_aligner_weights(self, fpath):
        self.text_aligner.save_weights(fpath)

    def train_mode(
        self,
        lr=.0002,
        loss_margin=0.2,
        clip_grad_value=2.
    ):
        # create full VSE++ model and optimizer
        img_aligner_input = tf.keras.layers.Input(
            shape=(self.img_encoder.embedding_dim,)
        )
        text_aligner_input = tf.keras.layers.Input(
            shape=(self.text_encoder.embedding_dim,)
        )
        aligned_img = self.img_aligner(img_aligner_input)
        aligned_text = self.text_aligner(text_aligner_input)
        concat = tf.keras.layers.Concatenate()([aligned_img, aligned_text])
        self.vse_model = tf.keras.Model(
            [img_aligner_input, text_aligner_input],
            concat
        )
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.loss_margin = loss_margin
        self.clip_grad_value = clip_grad_value

    def train_aligners_step(self, img_vectors, text_vectors):
        with tf.GradientTape() as tape:
            embeddings = self.vse_model(
                [img_vectors, text_vectors],
                training=True
            )
            img_embed, text_embed = tf.split(embeddings, 2, axis=1)
            loss = vsepp_mh_loss(
                img_embed,
                text_embed,
                margin=self.loss_margin
            )
        grads = tape.gradient(loss, self.vse_model.trainable_variables)
        grads = [
            tf.clip_by_value(grad, -self.clip_grad_value, self.clip_grad_value)
            for grad in grads
        ]
        self.optimizer.apply_gradients(
            zip(grads, self.vse_model.trainable_variables)
        )

        embeddings = self.vse_model(
            [img_vectors, text_vectors],
            training=False
        )
        img_embed, text_embed = tf.split(embeddings, 2, axis=1)
        return loss

    def evaluate_aligners(self, img_vectors, text_vectors):
        embeddings = self.vse_model.predict(
            [img_vectors, text_vectors]
        )
        img_embs, text_embs = tf.split(embeddings, 2, axis=1)
        loss = vsepp_mh_loss(img_embs, text_embs)
        img_embs = img_embs.numpy()
        text_embs = text_embs.numpy()
        img2text_recall_k1 = utils.recall(img_embs, text_embs, k=1)
        img2text_recall_k5 = utils.recall(img_embs, text_embs, k=5)
        img2text_recall_k10 = utils.recall(img_embs, text_embs, k=10)

        text2img_recall_k1 = utils.recall(text_embs, img_embs, k=1)
        text2img_recall_k5 = utils.recall(text_embs, img_embs, k=5)
        text2img_recall_k10 = utils.recall(text_embs, img_embs, k=10)
        return (loss,
                img2text_recall_k1,
                img2text_recall_k5,
                img2text_recall_k10,
                text2img_recall_k1,
                text2img_recall_k5,
                text2img_recall_k10)

    def embed_images(self, img_paths, batch_size=512, align=True):
        return self.img_aligner.predict(
            self.img_encoder.embed(img_paths)
            )

    def embed_texts(self, texts, batch_size=512, align=True):
        return self.text_aligner.predict(
            self.text_encoder.embed(texts)
        )

    def embed(
        self,
        img_paths,
        texts,
        img_merge_strategy='avg',
        text_merge_strategy='avg',
        multimodal_merge_strategy='avg'
    ):
        if len(img_paths):
            img_vectors = self.embed_images(img_paths)

            if img_merge_strategy == 'avg':
                img_vector = np.mean(img_vectors, axis=0)
            else:
                raise NotImplementedError
        else:
            img_vector = None

        if len(texts):
            text_vectors = self.embed_texts(texts)
            if text_merge_strategy == 'avg':
                text_vector = np.mean(text_vectors, axis=0)
            else:
                raise NotImplementedError
        else:
            text_vector = None

        if img_vector is not None and text_vector is not None:
            if multimodal_merge_strategy == 'avg':
                vec = np.mean(np.vstack(img_vector, text_vector), axis=0)
            return vec
        elif img_vector is not None:
            return img_vector
        elif text_vector is not None:
            return text_vector
        else:
            raise ValueError('Cannot create embedding: no content.')
