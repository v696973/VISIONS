# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import normalize


def chunks(l, n):
    '''
    split list l into chunks of size n
    '''
    return [l[i:i + n] for i in range(0, len(l), n)]


def l2norm(v):
    '''
    scale vectors to unit norm (vector length).
    '''
    return normalize(v, norm='l2')


def tf_l2norm(X):
    norm = tf.sqrt(tf.reduce_sum(tf.pow(X, 2), axis=1, keepdims=True))
    X = tf.divide(X, norm)
    return X


def recall(a, b, k=1):
    assert a.shape == b.shape
    scores = np.dot(a, b.T)
    total_recall = 0
    for i in range(a.shape[0]):
        scores_i = np.argsort(scores[i])[::-1][:k]
        if i in scores_i:
            total_recall += 1

    return total_recall / a.shape[0]


def is_binary_file(fname):
    # based on https://stackoverflow.com/questions/898669#7392391
    with open(fname, 'rb') as f:
        s = f.read(1024)
    # if type(s) is not bytes:
    #     s = s.encode('utf-8', errors='ignore')
    textchars = bytearray(
        {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f}
    )
    return bool(s.translate(None, textchars))


def img_or_text(fname):
    try:
        img = Image.open(fname)
        img.verify()
        return 'img'
    except Exception:  # I don't really know what exactly will 'verify()' rise
        pass

    if is_binary_file(fname):
        return None
    else:  # let's be optimistic
        return 'text'
