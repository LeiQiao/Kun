from . import CPPDataCoder
from . import CPPDataType

import random


# c++ convolution code generator
class CPPOpCoder_Activation:
    name = ""
    input_shape = None
    output_shape = None
    dim_ordering = None
    type = CPPDataType.ACTIVATION_LINEAR

    # initialize with activation type
    def __init__(self, input_shape, dim_ordering, act_type):
        if act_type == CPPDataType.ACTIVATION_LINEAR:
            self.name = "linear"
        elif act_type == CPPDataType.ACTIVATION_RELU:
            self.name = "relu"
        elif act_type == CPPDataType.ACTIVATION_TANH:
            self.name = "tanh"
        elif act_type == CPPDataType.ACTIVATION_SOFT_MAX:
            self.name = "softmax"
            assert len(input_shape) == 1, "SoftMax op must has one dimension"
        else:
            assert False, "Unsupport activation type: "+str(act_type)
        
        self.name = self.name+str(random.randint(10000, 100000))

        self.input_shape = input_shape
        self.output_shape = list(input_shape)
        self.dim_ordering = dim_ordering
        self.type = act_type

    # dump activation operation to c++ file
    def dump_to_file(self, file, data_coder, cpp_type):
        template_code = ""
        if self.type == CPPDataType.ACTIVATION_LINEAR:
            template_code = open('templates/linear.tmp').read()
        elif self.type == CPPDataType.ACTIVATION_RELU:
            template_code = open('templates/relu.tmp').read()
        elif self.type == CPPDataType.ACTIVATION_TANH:
            template_code = open('templates/tanh.tmp').read()
        elif self.type == CPPDataType.ACTIVATION_SOFT_MAX:
            template_code = open('templates/softmax.tmp').read()

        template_code = template_code.replace('%NAME%', self.name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(cpp_type))
        template_code = template_code.replace('%DIMENSION_COUNTS%', str(len(self.input_shape)))
        template_code = template_code.replace('%SOFTMAX_LENGTH%', str(self.input_shape[0]))

        file.write(template_code)
