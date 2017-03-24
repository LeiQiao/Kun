from Keras.keras2cpp import keras2cpp
from UnitTest.UTProjBuilder import UTProjBuilder
import numpy as np


k2c = keras2cpp("/root/zuoye/model/model_structure_number.json", "/root/zuoye/model/model_weight_number.h5", None, "idcard_number")
ut = UTProjBuilder(k2c)
ut.build_project()

tensor = np.zeros((1, 32, 32, 1))
ut.test(tensor)
tensor = np.ones((1, 32, 32, 1))
ut.test(tensor)
ut.remove_project()
quit()

roadmap = CPPCoderRoadMap.CPPCoderRoadMap("model_test")

file = open('test.cpp', 'w')

kernel = np.array([[[[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]]]])
bias = np.array([1.,2.,3.,4.,5.])
conv = CPPOpCoder_Conv2D.CPPOpCoder_Conv2D([28,28,2], CPPDataType.ORDER_NHWC, kernel, bias, 1, 1, 0)
roadmap.append_op(conv)

linear = CPPOpCoder_Activation.CPPOpCoder_Activation([10, 10, 2], CPPDataType.ORDER_NHWC, CPPOpCoder_Activation.LINEAR)
roadmap.append_op(linear)

relu = CPPOpCoder_Activation.CPPOpCoder_Activation([10, 10, 2], CPPDataType.ORDER_NHWC, CPPOpCoder_Activation.RELU)
roadmap.append_op(relu)

tanh = CPPOpCoder_Activation.CPPOpCoder_Activation([10, 10, 2], CPPDataType.ORDER_NHWC, CPPOpCoder_Activation.TANH)
roadmap.append_op(tanh)

maxpool = CPPOpCoder_Pool.CPPOpCoder_Pool([10, 10, 2], CPPDataType.ORDER_NHWC, CPPOpCoder_Pool.MAX_POOL, [2, 2])
roadmap.append_op(maxpool)

flatten = CPPOpCoder_Flatten.CPPOpCoder_Flatten([5, 5, 2], CPPDataType.ORDER_NHWC)
roadmap.append_op(flatten)

weight = np.ones([50, 5], dtype="float32")
bias = np.array([1.,2.,3.,4.,5.])
dense = CPPOpCoder_Dense.CPPOpCoder_Dense([50], CPPDataType.ORDER_NHWC, weight, bias)
roadmap.append_op(dense)

softmax = CPPOpCoder_Activation.CPPOpCoder_Activation([5], CPPDataType.ORDER_NHWC, CPPOpCoder_Activation.SOFT_MAX)
roadmap.append_op(softmax)

roadmap.dump_codes("./", CPPDataType.FLOAT32)
