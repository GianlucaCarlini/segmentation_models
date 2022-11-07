import tensorflow as tf


EPSILON = tf.keras.backend.epsilon()


def mean_iou(y_true, y_pred, threshold=None):

    if threshold is not None:
        y_pred = tf.greater(y_pred, threshold)
        y_pred = tf.cast(y_pred, tf.float32())

    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum((y_true + y_pred), axis=[0, 1, 2]) - intersection

    score = tf.divide((intersection + EPSILON), (union + EPSILON))

    return tf.reduce_mean(score)


def f1_score(y_true, y_pred, threshold=None, beta=1):

    if threshold is not None:
        y_pred = tf.greater(y_pred, threshold)
        y_pred = tf.cast(y_pred, tf.float32())

    tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    fp = tf.reduce_sum(y_pred, axis=[0, 1, 2]) - tp
    fn = tf.reduce_sum(y_true, axis=[0, 1, 2]) - tp

    score = ((1 + beta**2) * tp + EPSILON) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + EPSILON
    )
    score = tf.reduce_mean(score)

    return score
