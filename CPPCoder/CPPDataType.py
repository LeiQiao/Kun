# c++ type define
INT8 = 1
INT16 = 2
INT32 = 3
INT64 = 4

UINT8 = 5
UINT16 = 6
UINT32 = 8
UINT64 = 9

FLOAT32 = 10
FLOAT64 = 11

# input data order type, N: number H: height W:width C:channel
ORDER_NHWC = 0
ORDER_NCHW = 1

# activation type
ACTIVATION_LINEAR = 0
ACTIVATION_RELU = 1
ACTIVATION_TANH = 2
ACTIVATION_SOFT_MAX = 3

# pool type
POOL_MAX_POOL = 0

# bits of c++ type
def bits(cpp_type):
    if cpp_type == INT8:
        return 8
    elif cpp_type == INT16:
        return 16
    elif cpp_type == INT32:
        return 32
    elif cpp_type == INT64:
        return 64
    elif cpp_type == UINT8:
        return 8
    elif cpp_type == UINT16:
        return 16
    elif cpp_type == UINT32:
        return 32
    elif cpp_type == UINT64:
        return 64
    elif cpp_type == FLOAT32:
        return 32
    elif cpp_type == FLOAT64:
        return 64
    else: assert False, ("Unsupported data type: "+str(scpp_type))

# python struct.pack type of c++ type
def pack_type(cpp_type):
    if cpp_type == INT8:
        return "b"
    elif cpp_type == INT16:
        return "h"
    elif cpp_type == INT32:
        return "i"
    elif cpp_type == INT64:
        return "l"
    elif cpp_type == UINT8:
        return "B"
    elif cpp_type == UINT16:
        return "H"
    elif cpp_type == UINT32:
        return "I"
    elif cpp_type == UINT64:
        return "L"
    elif cpp_type == FLOAT32:
        return "f"
    elif cpp_type == FLOAT64:
        return "d"
    else: assert False, ("Unsupported data type: "+str(scpp_type))

def type_string(cpp_type):
    if cpp_type == INT8:
        return "int8_t"
    elif cpp_type == INT16:
        return "int16_t"
    elif cpp_type == INT32:
        return "int32_t"
    elif cpp_type == INT64:
        return "int64_t"
    elif cpp_type == UINT8:
        return "uint8_t"
    elif cpp_type == UINT16:
        return "uint16_t"
    elif cpp_type == UINT32:
        return "uint32_t"
    elif cpp_type == UINT64:
        return "uint64_t"
    elif cpp_type == FLOAT32:
        return "float"
    elif cpp_type == FLOAT64:
        return "double"
    else: assert False, ("Unsupported data type: "+str(scpp_type))

def dtype2cpptype(dtype):
    if dtype == "int8":
        return INT8
    elif dtype == "int16":
        return INT16
    elif dtype == "int32":
        return INT32
    elif dtype == "int64":
        return INT64
    elif dtype == "float32":
        return FLOAT32
    elif dtype == "float64":
        return FLOAT64

def dim_order_from_keras(dim_ordering):
    if dim_ordering == 'tf':
        return ORDER_NHWC
    elif dim_ordering == 'th':
        return ORDER_NCHW
    else:
        assert False, "Unsupport dim_ordering: "+dim_ordering

def activation_type_from_keras(keras_activation):
    if keras_activation == "linear":
        return ACTIVATION_LINEAR
    elif keras_activation == "relu":
        return ACTIVATION_RELU
    elif keras_activation == "softmax":
        return ACTIVATION_SOFT_MAX
    elif keras_activation == "tanh":
       return ACTIVATION_TANH
    else:
        assert False, "Unsupported activation type: %s" % act