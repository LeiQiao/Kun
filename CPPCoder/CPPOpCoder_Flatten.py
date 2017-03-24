from . import CPPDataCoder
from . import CPPDataType

import random

# c++ flatten code generator
class CPPOpCoder_Flatten:
    name = ""
    input_shape = None
    output_shape = None
    dim_ordering = None

    # initialize
    def __init__(self, input_shape, dim_ordering):
        self.name = "flatten"+str(random.randint(10000, 100000))
        self.input_shape = input_shape
        self.output_shape = [1]
        for index in range(len(self.input_shape)):
            self.output_shape[0] = self.output_shape[0] * int(self.input_shape[index])
        self.dim_ordering = dim_ordering

    # dump flatten operation to c++ file
    def dump_to_file(self, file, data_coder, cpp_type):
        template_code = open('templates/flatten.tmp').read()

        template_code = template_code.replace('%NAME%', self.name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(cpp_type))
        template_code = template_code.replace('%DIMENSION_COUNTS%', str(len(self.input_shape)))
        template_code = template_code.replace('%OUTPUT_LENGTH%', str(self.output_shape[0]))

        file.write(template_code)
