import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize


def chunks(l, n):
    '''
    split list l into chunks of size n
    '''
    return [l[i:i + n] for i in range(0, len(l), n)]


def l2norm(vector):
    '''
    scale vector to unit norm (vector length).
    '''
    return normalize([vector], norm='l2')[0]


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
