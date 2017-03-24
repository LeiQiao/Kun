from Keras.keras2cpp import keras2cpp
from UnitTest.UTProjBuilder import UTProjBuilder
import numpy as np

# load keras model
k2c = keras2cpp("./test/model_structure.json", "./test/model_weight.h5", None, "mnist")

# build unit test
ut = UTProjBuilder(k2c)
ut.build_project()

# unit test 1
tensor = np.zeros((1, 28, 28, 1))
ut.test(tensor)

# unit test 2
tensor = np.ones((1, 28, 28, 1))
ut.test(tensor)

# remove unit test
ut.remove_project()

# dump cpp files from keras model
k2c.save_to_path("./test/cpp/")

quit()
