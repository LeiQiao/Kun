from . import CPPDataCoder
from . import CPPDataType

import random


# c++ pool code generator
class CPPOpCoder_Pool:
    name = ""
    input_shape = None
    output_shape = None
    dim_ordering = None
    type = CPPDataType.POOL_MAX_POOL
    pool_size = []

    # initialize with activation type
    def __init__(self, input_shape, dim_ordering, act_type, pool_size):
        assert len(input_shape) == 3, "input shape must have 3 dimensions, HWC or CHW"
        assert len(pool_size) == 2, "pool size error"

        if act_type == CPPDataType.POOL_MAX_POOL:
            self.name = "maxpool"
        else:
            assert False, "Unsupport pool type: "+act_type
        self.name = self.name+str(random.randint(10000, 100000))
        self.input_shape = input_shape
        self.dim_ordering = dim_ordering
        self.type = act_type
        self.pool_size = pool_size
        self.output_shape = list(self.input_shape)

        if dim_ordering == CPPDataType.ORDER_NCHW:
            self.output_shape[1] = int(self.input_shape[1] / self.pool_size[0])
            self.output_shape[2] = int(self.input_shape[2] / self.pool_size[1])
        elif dim_ordering == CPPDataType.ORDER_NHWC:
            self.output_shape[0] = int(self.input_shape[0] / self.pool_size[0])
            self.output_shape[1] = int(self.input_shape[1] / self.pool_size[1])


    # dump activation operation to c++ file
    def dump_to_file(self, file, data_coder, cpp_type):
        template_code = ""
        if self.type == CPPDataType.POOL_MAX_POOL:
            if self.dim_ordering == CPPDataType.ORDER_NCHW:
                template_code = open('templates/maxpool_NCHW.tmp').read()
            elif self.dim_ordering == CPPDataType.ORDER_NHWC:
                template_code = open('templates/maxpool_NHWC.tmp').read()

        template_code = template_code.replace('%NAME%', self.name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(cpp_type))
        template_code = template_code.replace('%DIMENSION_COUNTS%', str(len(self.input_shape)))
        template_code = template_code.replace('%POOL_SIZE_ROWS%', str(self.pool_size[0]))
        template_code = template_code.replace('%POOL_SIZE_COLS%', str(self.pool_size[1]))

        if self.dim_ordering == CPPDataType.ORDER_NCHW:
            template_code = template_code.replace('%INPUT_CHANNELS%', str(self.input_shape[0]))
            template_code = template_code.replace('%INPUT_ROWS%', str(self.input_shape[1]))
            template_code = template_code.replace('%INPUT_COLS%', str(self.input_shape[2]))
            template_code = template_code.replace('%OUTPUT_ROWS%', str(self.output_shape[1]))
            template_code = template_code.replace('%OUTPUT_COLS%', str(self.output_shape[2]))
        if self.dim_ordering == CPPDataType.ORDER_NHWC:
            template_code = template_code.replace('%INPUT_CHANNELS%', str(self.input_shape[2]))
            template_code = template_code.replace('%INPUT_ROWS%', str(self.input_shape[0]))
            template_code = template_code.replace('%INPUT_COLS%', str(self.input_shape[1]))
            template_code = template_code.replace('%OUTPUT_ROWS%', str(self.output_shape[0]))
            template_code = template_code.replace('%OUTPUT_COLS%', str(self.output_shape[1]))

        file.write(template_code)
