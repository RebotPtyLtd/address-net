from typing import Dict, Optional

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_estimator as tf_estimator

from addressnet.dataset import vocab, n_labels


def model_fn(features: Dict[str, tf.Tensor], labels: tf.Tensor, mode: str, params) -> tf_estimator.estimator.EstimatorSpec:
    """
    The AddressNet model function suitable for tf_estimator.estimator.Estimator
    :param features: a dictionary containing tensors for the encoded_text and lengths
    :param labels: a label for each character designating its position in the address
    :param mode: indicates whether the model is being trained, evaluated or used in prediction mode
    :param params: model hyperparameters, including rnn_size and rnn_layers
    :return: the appropriate tf_estimator.estimator.EstimatorSpec for the model mode
    """
    encoded_text, lengths = features['encoded_text'], features['lengths']
    rnn_size = params.get("rnn_size", 128)
    rnn_layers = params.get("rnn_layers", 3)

    embeddings = tf.get_variable("embeddings", dtype=tf.float32, initializer=tf.random_normal(shape=(len(vocab), 8)))
    encoded_strings = tf.nn.embedding_lookup(embeddings, encoded_text)

    logits, loss = nnet(encoded_strings, lengths, rnn_layers, rnn_size, labels, mode == tf_estimator.estimator.ModeKeys.TRAIN)

    predicted_classes = tf.argmax(logits, axis=2)

    if mode == tf_estimator.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.softmax(logits)
        }
        return tf_estimator.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf_estimator.estimator.ModeKeys.EVAL:
        metrics = {}
        return tf_estimator.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf_estimator.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=tf.train.get_global_step())
        return tf_estimator.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def nnet(encoded_strings: tf.Tensor, lengths: tf.Tensor, rnn_layers: int, rnn_size: int, labels: tf.Tensor = None,
         training: bool = True) -> (tf.Tensor, Optional[tf.Tensor]):
    """
    Generates the RNN component of the model
    :param encoded_strings: a tensor containing the encoded strings (embedding vectors)
    :param lengths: a tensor of string lengths
    :param rnn_layers: number of layers to use in the RNN
    :param rnn_size: number of units in each layer
    :param labels: labels for each character in the string (optional)
    :param training: if True, dropout will be enabled on the RNN
    :return: logits and loss (loss will be None if labels is not provided)
    """

    mask = tf.sequence_mask(lengths)
    rnn_output = encoded_strings

    for _ in range(rnn_layers):
        gru_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(rnn_size, return_sequences=True, dropout=0.2)
        )
        rnn_output = gru_layer(rnn_output, mask=mask, training=training)

    logits = tf.keras.layers.Dense(n_labels, activation=tf.nn.elu)(rnn_output)

    loss = None
    if labels is not None:
        mask_float = tf.cast(mask, tf.float32)
        loss = tf.losses.softmax_cross_entropy(labels, logits, weights=mask_float)
    return logits, loss
