import tempfile
import shutil
import urllib.request
import hashlib
import os
import gzip
import tarfile

import numpy as np

from CPPCoder import CPPDataType
from CPPCoder.CPPDataCoder import CPPDataCoder
from Keras.keras2cpp import keras2cpp
from keras2tensorflow.keras2tensorflow import keras2tensorflow

EIGEN_URL = "https://bitbucket.org/eigen/eigen/get/60578b474802.tar.gz"
EIGEN_SHA256 = "7527cda827aff351981ebd910012e16be4d899c28a9ae7f143ae60e7f3f7b83d"
EIGEN_STRIP_PREFIX = "eigen-eigen-60578b474802"

class UTProjBuilder:
    code_builder = None
    project_dir = ""
    tf_builder = None

    def __init__(self, code_builder):
        self.code_builder = code_builder
        self.project_dir = ""

    def build_project(self):
        self.project_dir = tempfile.mkdtemp("", "Kun")

        os.mkdir(self.project_dir+"/src/")
        self.code_builder.save_to_path(self.project_dir+"/src/")

        print("downloading Eigen ... ("+self.project_dir+"/eigen.tar.gz"+")")
        urllib.request.urlretrieve(EIGEN_URL, self.project_dir+"/eigen.tar.gz")
        # shutil.copy("/root/eigen.tar.gz", self.project_dir+"/eigen.tar.gz")
        f = open(self.project_dir+"/eigen.tar.gz", 'rb')
        sh = hashlib.sha256()
        sh.update(f.read())
        if sh.hexdigest() != EIGEN_SHA256:
            print("error Eigen check error (sha256=\""+sh.hesdigest()+"\" expect \""+EIGEN_SHA256+"\")")
            return

        print("unpack Eigen ... ("+self.project_dir+"/supports/)")
        g_file = gzip.GzipFile(self.project_dir+"/eigen.tar.gz")
        open(self.project_dir+"/eigen.tar", "wb").write(g_file.read())
        g_file.close()
        os.remove(self.project_dir+"/eigen.tar.gz")

        tar = tarfile.open(self.project_dir+"/eigen.tar")
        names = tar.getnames()
        for name in names:
            tar.extract(name, self.project_dir)
        tar.close()
        os.remove(self.project_dir+"/eigen.tar")
        os.rename(self.project_dir+"/"+EIGEN_STRIP_PREFIX, self.project_dir+"/eigen")

        print("copying support files ... ("+self.project_dir+"/supports/)")
        shutil.copytree("UnitTest/EigenCustomCode", self.project_dir+"/supports")

        print("generating CMakeLists.txt ... ("+self.project_dir+"/CMakeLists.txt)")
        shutil.move(self.project_dir+"/supports/CMakeLists.txt", self.project_dir+"/CMakeLists.txt")

        template_code = open(self.project_dir+"/CMakeLists.txt").read()
        template_code = template_code.replace('%MODEL_NAME%', self.code_builder.road_map.model_name)
        f = open(self.project_dir+"/CMakeLists.txt", "w")
        f.write(template_code)
        f.close()

    def remove_project(self):
        shutil.rmtree(self.project_dir)

    def test(self, tensor):
        tensor_shape = []
        for dim in range(len(tensor.shape)-1):
            tensor_shape.append(tensor.shape[dim+1])
        expect_input_shape = self.code_builder.road_map.ops[0].input_shape

        if tensor_shape != expect_input_shape:
            print("unable run test: input shape is "+str(tensor_shape)+" expect input shape is "+str(expect_input_shape))
            return

        # write test data file
        data_coder = CPPDataCoder("unit_test_data")
        data_coder.append("unit_test_data", tensor.flatten(), self.code_builder.cpp_type)
        data_coder.dump_to_file(self.project_dir+"/unit_test_data.hpp")

        # write main file
        shutil.copy(self.project_dir+"/supports/main.cpp", self.project_dir+"/main.cpp")
        template_code = open(self.project_dir+"/main.cpp").read()
        template_code = template_code.replace('%MODEL_NAME%', self.code_builder.road_map.model_name)
        template_code = template_code.replace('%CPP_TYPE%', CPPDataType.type_string(self.code_builder.cpp_type))
        template_code = template_code.replace('%DIM_SIZE_1%', str(expect_input_shape[0]))
        template_code = template_code.replace('%DIM_SIZE_2%', str(expect_input_shape[1]))
        template_code = template_code.replace('%DIM_SIZE_3%', str(expect_input_shape[2]))
        file = open(self.project_dir+"/main.cpp", "w")
        file.write(template_code)
        file.close()

        # build project
        print("build project ...")
        if os.path.exists(self.project_dir+"/build/"): shutil.rmtree(self.project_dir+"/build/")
        os.mkdir(self.project_dir+"/build/")
        pwd = os.getcwd()
        os.chdir(self.project_dir+"/build/")
        os.system("cmake ..")
        os.system("make")
        print("run test ...")
        os.system("./"+self.code_builder.road_map.model_name+"")
        os.chdir(pwd)

        # load every step's result
        tf_builder = keras2tensorflow(self.code_builder.keras_model)
        for layer_index in range(tf_builder.layer_count()):
            print("checking layer outputs: "+type(self.code_builder.keras_model.layers[layer_index]).__name__)

            # load tensorflow result
            tf_result = tf_builder.predict_step(tensor, layer_index).flatten()

            # load cpp result
            op = self.code_builder.road_map.ops[layer_index]
            print("output shape: "+str(op.output_shape))
            cpp_result = []
            print(op.name)
            if len(op.output_shape) == 1:
                cpp_result = self.read_floats_from_file(self.project_dir+"/build/"+op.name+"/"+op.name)
            elif len(op.output_shape) == 3:
                if op.dim_ordering == CPPDataType.ORDER_NHWC:
                    floats_np = np.zeros(op.output_shape)
                    for i in range(op.output_shape[2]):
                        floats = self.read_floats_from_file(self.project_dir+"/build/"+op.name+"/"+op.name+"_"+str(i+1))
                        floats_np[:, :, i] = np.array(floats).reshape(op.output_shape[0], op.output_shape[1])
                    cpp_result = floats_np.flatten()
                elif op.dim_ordering == CPPDataType.ORDER_NCHW:
                    floats_np = np.zeros(op.output_shape)
                    for i in range(op.output_shape[0]):
                        floats = self.read_floats_from_file(self.project_dir+"/build/"+op.name+"/"+op.name+"_"+str(i+1))
                        floats_np[i, :, :] = np.array(floats).reshape(op.output_shape[1], op.output_shape[2])
                    cpp_result = floats_np.flatten()

            # compare tensorflow & cpp results
            assert len(tf_result) == len(cpp_result), "error cpp has "+str(len(cpp_result))+" outputs expect ("+len(tf_result)+")"

            check_diff_pass = True
            for index in range(len(tf_result)):
                diff = tf_result[index] - cpp_result[index]
                if abs(diff) > 0.0001:
                    print("index ("+str(index)+"): tensorflow output: ("+str(tf_result[index])+") cpp output: ("+str(cpp_result[index])+") diff: "+str(diff))
                    check_diff_pass = False
            
            if not check_diff_pass: return False
        
        print("unit test passed, all goods!!!!!!!!!!!!!\n")


        # read test result
        # cpp_input = []
        # for value in open(self.project_dir+"/build/unit_test_input.txt", "r").read().split(","):
        #     if len(value) == 0: continue
        #     cpp_input.append(float(value))
        # print(cpp_input)

        # cpp_result = []
        # for result in open(self.project_dir+"/build/unit_test_result.txt", "r").read().split(","):
        #     if len(result) == 0: continue
        #     print(result)
        #     cpp_result.append(float(result))
        # print(cpp_result)


        # # run keras
        # keras_result = self.code_builder.keras_model.predict(tensor)
        # print(keras_result)

        # # run keras2tensorflow
        # k2t = keras2tensorflow(self.code_builder.keras_model)
        # tensorflow_result = k2t.prediction(tensor)
        # print(tensorflow_result)

    def read_floats_from_file(self, filename):
        floats = []
        for value in open(filename, "r").read().replace("\n", "").split(","):
            value = value.strip(' \t\n\r')
            if len(value) == 0: continue
            floats.append(float(value))
        return floats

