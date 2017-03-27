from . import CPPDataCoder
from . import CPPDataType

import random

# c++ flatten code generator
class CPPOpCoder_Dense:
    name = ""
    input_shape = None
    output_shape = None
    dim_ordering = None
    weight_data = []
    bias_data = []

    # initialize
    def __init__(self, input_shape, dim_ordering, weight_data, bias_data):
        assert len(weight_data.shape) == 2, "weight data must has 2 dimensions"
        assert weight_data.shape[0] == input_shape[0], "first dimension length of weight must == input length"
        assert weight_data.shape[1] == len(bias_data), "second dimension length of weight must == bias_data length"

        self.name = "dense"+str(random.randint(10000, 100000))
        self.input_shape = input_shape
        self.dim_ordering = dim_ordering
        self.weight_data = weight_data
        self.bias_data = bias_data
        self.output_shape = [len(bias_data)]

    # dump flatten operation to c++ file
    def dump_to_file(self, file, data_coder, cpp_type):
        template_code = ""
        template_code = open('templates/dense.tmp').read()

        template_code = template_code.replace('%NAME%', self.name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(cpp_type))
        template_code = template_code.replace('%INPUT_LENGTH%', str(self.weight_data.shape[0]))
        template_code = template_code.replace('%OUTPUT_LENGTH%', str(self.output_shape[0]))

        # save weight data
        weight_data_name = "data_weight_"+self.name
        data_coder.append(weight_data_name, self.weight_data.flatten(), cpp_type)

        # save bias data
        bias_data_name = "data_bias_"+self.name
        data_coder.append(bias_data_name, self.bias_data, cpp_type)

        template_code = template_code.replace('%WEIGHT_DATA_NAME%', weight_data_name)
        template_code = template_code.replace('%BIAS_DATA_NAME%', bias_data_name)

        file.write(template_code)
