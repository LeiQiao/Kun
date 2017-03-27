import tensorflow as tf
import numpy as np
import struct

LAYER_CONV2D = 1
LAYER_ACTIVATION = 2
LAYER_MAXPOOL = 3
LAYER_FLATTEN = 4
LAYER_DENSE = 5
LAYER_DROPOUT = 6

ACTIVATION_UNKNOWN = 0
ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTMAX = 3
ACTIVATION_TANH = 4

#####################################################################
class keras_conv2d:
    weights = None
    biases = None

    def __init__(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]
        self.biases = keras_layer.get_weights()[1]
        self.padding = 'VALID'
        if keras_layer.border_mode == "same":
            self.padding = 'SAME'

    def dump_tf_layer(self, prev_tf_layer):
        w = tf.constant(self.weights)
        b = tf.constant(self.biases)
        tf_layer = tf.nn.conv2d(prev_tf_layer,
                                w,
                                strides=[1,1,1,1],
                                padding=self.padding) + b
        return tf_layer

#####################################################################
class keras_activation:
    activation = ACTIVATION_UNKNOWN

    def __init__(self, keras_layer):
        act = keras_layer.get_config()['activation']
        if act == "linear":
            self.activation = ACTIVATION_LINEAR
        elif act == "relu":
            self.activation = ACTIVATION_RELU
        elif act == "softmax":
            self.activation = ACTIVATION_SOFTMAX
        elif act == "tanh":
            self.activation = ACTIVATION_TANH
        else:
            assert False, "Unsupported activation type: %s" % act
        

    def dump_tf_layer(self, prev_tf_layer):
        if self.activation == ACTIVATION_LINEAR:
            tf_layer = prev_tf_layer
        elif self.activation == ACTIVATION_RELU:
            tf_layer = tf.nn.relu(prev_tf_layer)
        elif self.activation == ACTIVATION_SOFTMAX:
            tf_layer = tf.nn.softmax(prev_tf_layer)
        elif self.activation == ACTIVATION_TANH:
            tf_layer = tf.tanh(prev_tf_layer)
        return tf_layer

#####################################################################
class keras_maxpool:
    pool_size = None
    padding = None

    def __init__(self, keras_layer):
        self.pool_size = keras_layer.get_config()['pool_size']
        self.padding = 'VALID'
        if keras_layer.border_mode != "valid":
            assert False, "Unsupported padding type: %s" % keras_layer.border_mode

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.nn.max_pool(prev_tf_layer,
                                  ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                                  strides=[1, self.pool_size[0], self.pool_size[1], 1],
                                  padding=self.padding)
        return tf_layer

#####################################################################
class keras_flatten:
    def __init__(self, keras_layer):None

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.reshape(prev_tf_layer, [-1])
        return tf_layer

#####################################################################
class keras_dense:
    weights = None
    biases = None

    def __init__(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]
        self.biases = keras_layer.get_weights()[1]

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.reshape(prev_tf_layer, [-1, self.weights.shape[0]])
        tf_layer = tf.matmul(tf_layer, self.weights) + self.biases
        tf_layer = tf.reshape(tf_layer, [-1])
        return tf_layer

#####################################################################
class keras_dropout:
    p = 0
    
    def __init__(self, keras_layer):
        self.p = keras_layer.p

    def dump_tf_layer(self, prev_tf_layer):
        # prob = tf.constant(self.p)
        prob = tf.constant(1.0)
        tf_layer = tf.nn.dropout(prev_tf_layer, prob)
        return tf_layer

#####################################################################
class keras2tensorflow:
    layers = []
    input_shape = []

    def __init__(self, keras_model):
        self.layers = []
        self.input_shape = keras_model.layers[0].batch_input_shape
        for keras_layer in keras_model.layers:
            layer_type = type(keras_layer).__name__

            tf_layer = None
            if layer_type == "Convolution2D":
                tf_layer = keras_conv2d(keras_layer)
            elif layer_type == "Activation":
                tf_layer = keras_activation(keras_layer)
            elif layer_type == "MaxPooling2D":
                tf_layer = keras_maxpool(keras_layer)
            elif layer_type == "Flatten":
                tf_layer = keras_flatten(keras_layer)
            elif layer_type == "Dense":
                tf_layer = keras_dense(keras_layer)
            elif layer_type == "Dropout":
                tf_layer = keras_dropout(keras_layer)
            else:
                assert False, "Unsupported layer type: %s" % layer_type
            
            self.layers.append(tf_layer)

    def save_protobuf(self, filename):
        graph_dump = tf.Graph()
        with graph_dump.as_default():
            tf_input = tf.placeholder("float32", self.input_shape, name="input")
            tf_prediction = self.dump_tf_layer(tf_input)
            tf_output = tf.add(tf_prediction, 0, name="output")

            sess = tf.Session()
            graph_def = graph_dump.as_graph_def()
            tf.train.write_graph(graph_def, '', filename, as_text=False)
            sess.close()

    def layer_count(self):
        return len(self.layers)

    def dump_tf_layer_step(self, prev_tf_layer, index):
        if (index < 0) or (index >= len(self.layers)): index = len(self.layers)-1
        now = 0
        for tf_layer in self.layers:
            prev_tf_layer = tf_layer.dump_tf_layer(prev_tf_layer)
            now += 1
            if now > index: break
        return prev_tf_layer

    def predict_step(self, data, index):
        sess = tf.Session()
        tf_input = tf.placeholder("float32", self.input_shape, name="input")
        tf_predict = self.dump_tf_layer_step(tf_input, index)
        result = sess.run(tf_predict, feed_dict={tf_input:data})
        sess.close()
        return result

    def dump_tf_layer(self, prev_tf_layer):
        return self.dump_tf_layer_step(prev_tf_layer, -1)

    def predict(self, data):
        return self.predict_step(data, -1)

