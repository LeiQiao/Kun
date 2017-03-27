from Keras.keras2cpp import keras2cpp
from UnitTest.UTProjBuilder import UTProjBuilder
import numpy as np
import os
import shutil

# load keras model
k2c = keras2cpp("../zuoye/model/step1.json", "../zuoye/model/step1.h5", None, "idcard_step1")

# build unit test
ut = UTProjBuilder(k2c)
ut.build_project()

# unit test 1
tensor = np.zeros((1, 3, 30, 428))
ut.test(tensor)

# unit test 2
tensor = np.ones((1, 3, 30, 428))
ut.test(tensor)

# remove unit test
ut.remove_project()

# dump cpp files from keras model
if os.path.exists("./test/cpp/"): shutil.rmtree("./test/cpp/")
os.mkdir("./test/cpp/")
k2c.save_to_path("./test/cpp/")

quit()
