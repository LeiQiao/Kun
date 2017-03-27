from . import CPPDataCoder
from . import CPPDataType

import random

# c++ convolution code generator
class CPPOpCoder_Conv2D:
    name = ""
    input_shape = None
    output_shape = None
    dim_ordering = None
    kernel_data = None
    bias_data = None
    stride_x = 1
    stride_y = 1
    border_type = CPPDataType.BORDER_VALID
    zero_padding = 0

    # initialize with convolution attributes
    def __init__(self, input_shape, dim_ordering, kernel_data, bias_data, stride_x, stride_y, border_type):
        assert len(input_shape) == 3, "input shape must have 3 dimensions, HWC or CHW"
        assert len(kernel_data.shape) == 4, "kernel shape must have 4 dimensions"
        assert stride_x > 0 and stride_y > 0, "stride must > 0"

        input_channel_count = 0
        kernel_channel_count = 0

        bias_count = len(bias_data)
        kernel_count = 0

        kernel_width = 0
        kernel_height = 0

        if dim_ordering == CPPDataType.ORDER_NHWC:
            kernel_channel_count = kernel_data.shape[2]
            kernel_count = kernel_data.shape[3]
            kernel_width = kernel_data.shape[0]
            kernel_height = kernel_data.shape[1]
            input_channel_count = input_shape[2]
        elif dim_ordering == CPPDataType.ORDER_NCHW:
            kernel_channel_count = kernel_data.shape[1]
            kernel_count = kernel_data.shape[0]
            kernel_width = kernel_data.shape[2]
            kernel_height = kernel_data.shape[3]
            input_channel_count = input_shape[0]

        assert bias_count == kernel_count, "bias counts("+str(bias_count)+") must == kernel filter counts("+str(kernel_count)+")"
        assert input_channel_count == kernel_channel_count, "input channels("+str(input_channel_count)+") != kernel channels("+str(kernel_channel_count)+")"

        self.name = "conv"+str(random.randint(10000, 100000))

        self.input_shape = input_shape
        self.dim_ordering = dim_ordering
        self.kernel_data = kernel_data
        self.bias_data = bias_data
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.border_type = border_type

        # output shape & dim ordering
        output_width = 0
        output_height = 0
        self.output_shape = list(input_shape)
        if dim_ordering == CPPDataType.ORDER_NCHW:
            output_width = input_shape[2]
            output_height = input_shape[1]
        elif self.dim_ordering == CPPDataType.ORDER_NHWC:
            output_width = input_shape[1]
            output_height = input_shape[0]

        if self.border_type == CPPDataType.BORDER_VALID:
            output_width -= kernel_width >> 1 << 1
            output_height -= kernel_height >> 1 << 1
            output_width = output_width / stride_x
            output_height = output_height / stride_y
            if output_width > int(output_width): output_width = int(output_width) + 1
            if output_height > int(output_height): output_height = int(output_height) + 1
        else:
            self.zero_padding = kernel_width >> 1

        if dim_ordering == CPPDataType.ORDER_NCHW:
            self.output_shape[2] = int(output_width)
            self.output_shape[1] = int(output_height)
            self.output_shape[0] = kernel_count
        elif self.dim_ordering == CPPDataType.ORDER_NHWC:
            self.output_shape[1] = int(output_width)
            self.output_shape[0] = int(output_height)
            self.output_shape[2] = kernel_count

    # dump convolve operation to c++ file
    def dump_to_file(self, file, data_coder, cpp_type):
        template_code = ""
        if self.dim_ordering == CPPDataType.ORDER_NCHW:
            template_code = open('templates/conv_NCHW.tmp').read()
        elif self.dim_ordering == CPPDataType.ORDER_NHWC:
            template_code = open('templates/conv_NHWC.tmp').read()

        template_code = template_code.replace('%NAME%', self.name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(cpp_type))
        template_code = template_code.replace('%STRIDE_X%', str(self.stride_x))
        template_code = template_code.replace('%STRIDE_Y%', str(self.stride_y))
        template_code = template_code.replace('%ZERO_PADDING%', str(self.zero_padding))
        if self.dim_ordering == CPPDataType.ORDER_NCHW:
            template_code = template_code.replace('%CHANNELS%', str(self.input_shape[0]))
            template_code = template_code.replace('%OUTPUT_ROWS%', str(self.output_shape[1]))
            template_code = template_code.replace('%OUTPUT_COLS%', str(self.output_shape[2]))
            template_code = template_code.replace('%KERNEL_ROWS%', str(self.kernel_data.shape[2]))
            template_code = template_code.replace('%KERNEL_COLS%', str(self.kernel_data.shape[3]))
            template_code = template_code.replace('%KERNEL_COUNTS%', str(self.kernel_data.shape[0]))
        elif self.dim_ordering == CPPDataType.ORDER_NHWC:
            template_code = template_code.replace('%CHANNELS%', str(self.input_shape[2]))
            template_code = template_code.replace('%OUTPUT_ROWS%', str(self.output_shape[0]))
            template_code = template_code.replace('%OUTPUT_COLS%', str(self.output_shape[1]))
            template_code = template_code.replace('%KERNEL_ROWS%', str(self.kernel_data.shape[0]))
            template_code = template_code.replace('%KERNEL_COLS%', str(self.kernel_data.shape[1]))
            template_code = template_code.replace('%KERNEL_COUNTS%', str(self.kernel_data.shape[3]))

        # save kernel data
        kernel_data_name = "data_kernel_"+self.name
        data_coder.append(kernel_data_name, self.kernel_data.flatten(), cpp_type)

        # save bias data
        bias_data_name = "data_bias_"+self.name
        data_coder.append(bias_data_name, self.bias_data, cpp_type)

        template_code = template_code.replace('%KERNEL_DATA_NAME%', kernel_data_name)
        template_code = template_code.replace('%BIAS_DATA_NAME%', bias_data_name)

        file.write(template_code)





