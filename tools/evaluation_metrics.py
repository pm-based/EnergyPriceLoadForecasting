import tensorflow as tf

def evaluation_metrics(_metrics):
    """
    Pass to the model compiler the required metics
    :param _metrics: generally are passed as self.settings['evaluation_metrics'] in the class fo the model
    :return: Array of metrics, they can be string (for the Keras implemented ones) or functions for the customized ones.
    """
    metrics = []

    for metric in _metrics:
        if metric == 'rmse':
            metrics.append(rmse)
        elif metric == 'smape':
            metrics.append(smape)
        elif metric == 'mae':
            metrics.append('mae')
        elif metric == 'mse':
            metrics.append('mse')
        else:
            print('Unknown metric: selected mse')
            metrics.append('mse')

    return metrics


def rmse(y_true, y_pred):
    y_pred = tf.squeeze(y_pred) # Is necessary to match tensor dimensions
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def smape(y_true, y_pred):
    y_pred = tf.squeeze(y_pred) # Is necessary to match tensor dimensions
    epsilon = tf.constant(1e-22) # To avoid denominator = 0
    denominator = 0.5*(tf.abs(y_true) + tf.abs(y_pred))
    return tf.reduce_mean(tf.abs(y_true - y_pred) / (denominator + epsilon))

