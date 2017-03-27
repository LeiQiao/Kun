from keras import backend as K
from keras.models import Sequential, model_from_json

import struct

from CPPCoder import CPPDataType
from CPPCoder.CPPDataCoder import CPPDataCoder
from CPPCoder.CPPOpCoder_Conv2D import CPPOpCoder_Conv2D
from CPPCoder.CPPOpCoder_Activation import CPPOpCoder_Activation
from CPPCoder.CPPOpCoder_Pool import CPPOpCoder_Pool
from CPPCoder.CPPOpCoder_Flatten import CPPOpCoder_Flatten
from CPPCoder.CPPOpCoder_Dense import CPPOpCoder_Dense
from CPPCoder.CPPCoderRoadMap import CPPCoderRoadMap

#####################################################################
class keras2cpp:
    road_map = None
    keras_model = None
    cpp_type = None

    def __init__(self, jsonfile, weightfile, mapfile, model_name):
        self.road_map = CPPCoderRoadMap(model_name)
        self.road_map.set_map(mapfile)

        self.keras_model = model_from_json(open(jsonfile).read())
        self.keras_model.load_weights(weightfile)

        self.cpp_type = CPPDataType.dtype2cpptype(self.keras_model.layers[0].input_dtype)

        dim_ordering = CPPDataType.dim_order_from_keras(self.keras_model.layers[0].dim_ordering)

        last_output_shape = list(self.keras_model.layers[0].input_shape)
        last_output_shape.pop(0)

        for keras_layer in self.keras_model.layers:
            layer_type = type(keras_layer).__name__
            print("start dump keras layer: "+layer_type)
            op = None

            if layer_type == "Convolution2D":
                kernels = keras_layer.get_weights()[0]
                biases = keras_layer.get_weights()[1]
                stride_x = keras_layer.subsample[0]
                stride_y = keras_layer.subsample[1]

                border_type = CPPDataType.BORDER_VALID
                if keras_layer.border_mode != "valid": border_type = CPPDataType.BORDER_SAME
                op = CPPOpCoder_Conv2D(last_output_shape, dim_ordering, kernels, biases, stride_x , stride_y, border_type)

            elif layer_type == "Activation":
                activation_type = CPPDataType.activation_type_from_keras(keras_layer.get_config()['activation'])
                op = CPPOpCoder_Activation(last_output_shape, dim_ordering, activation_type)

            elif layer_type == "MaxPooling2D":
                pool_size = keras_layer.get_config()['pool_size']
                op = CPPOpCoder_Pool(last_output_shape, dim_ordering, CPPDataType.POOL_MAX_POOL, pool_size)

            elif layer_type == "Flatten":
                op = CPPOpCoder_Flatten(last_output_shape, dim_ordering)

            elif layer_type == "Dense":
                weights = keras_layer.get_weights()[0]
                biases = keras_layer.get_weights()[1]
                op = CPPOpCoder_Dense(last_output_shape, dim_ordering, weights, biases)

            elif layer_type == "Dropout":
                continue

            else:
                assert False, "Unsupported layer type: %s" % layer_type
            
            print("layer input shape: "+str(op.input_shape))
            print("layer output shape: "+str(op.output_shape))
            print("")
            last_output_shape = op.output_shape
            self.road_map.append_op(op)

    def save_to_path(self, path):
        self.road_map.dump_codes(path, self.cpp_type)




