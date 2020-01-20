# -*- coding: utf-8 -*-
import os
import tensorflow as tf  # NOQA
import tensorflow_hub as hub
import tensorflow_text  # NOQA
import utils


class M_USEModel(object):

    def __init__(
        self,
        sentence_maxlen=10000,
        max_sentences=1000,
        tfhub_cache_dir='models/tf_hub'
    ):
        self.sentence_maxlen = sentence_maxlen
        self.max_sentences = max_sentences
        os.environ['TFHUB_CACHE_DIR'] = tfhub_cache_dir
        os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
        self.encoder = hub.load(
            'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'  # NOQA
        )
        self.embedding_dim = 512

    def preprocess_texts(self, texts):
        # TODO:
        # * split documents into sentences of length self.sentence_maxlen
        # ** think about a good way of splitting sentences/paragraphs?
        # * keep only self.max_sentences
        # * individual sentences should still be grouped by document

        # processed_texts = []
        # for text in texts:
        #     sentences = self.texts_to_sentences(
        #         text,
        #         sentence_maxlen=self.sentence_maxlen
        #     )
        #     sentences = sentences[:self.max_sentences]
        #     processed_texts.append(sentences)
        pass

    def embed(self, texts, l2_norm=False):
        vectors = self.encoder(texts).numpy()
        if l2_norm:
            vectors = utils.l2norm(vectors)

        return vectors
