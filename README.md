# Kun

A tool to transform from keras trained model to c++ code, it can generate small and fast c++ code, you can deploy it in everywhere (pc, MacOSX, linux, Android, iOS, even arm)

# Usage

You can find a fully functional demo in test.py

## dump keras model to cpp
```
# load keras model
# the 3rd parameter is a csv file to map the result indexs, the 4th parameter is the model name
k2c = keras2cpp("./test/mnist_keras_model/model_structure.json", "./test/mnist_keras_model/model_weight.h5", None, "mnist")

# dump cpp files
k2c.save_to_path("./test/cpp/")

```

## unit test
you can compare your cpp code with tensorflow to verify your code is correctly.
```
# build unit test
ut = UTProjBuilder(k2c)
ut.build_project()

tensor = ... # load test tensor, NHWC or NCHW ordering.
succ = ut.test(tensor)

# remove unit test
ut.remove_project()

```

## add cpp files to your own project
You can find a fully functional demo in test/main.cpp
1. keras2cpp will dump follow cpp files, add them to your project:

* \<model name\>.hpp & \<model name\>.cpp - your model's file
* \<model name\>_data.hpp - model's weights header
* \<model name\>_map.hpp - model's csv mapper header

2. copy the following file from "UnitTest/EigenCustomCode/" into your project:

* conv3x3_neon.hpp & conv3x3_neon.cpp
* convolve.hpp
* Eigen.h
* EigenCustomOp.hpp & EigenCustomOp.cpp

3. download the lastest eigen from eigen website, you should only specific eigen's path as include path.

4. compile it.

## predict

```
// write data to input tensor
Eigen::Tensor<float, 3, Eigen::RowMajor> input(image_height, image_width, 1);
...

// predict
Eigen::Tensor<float, 1, Eigen::RowMajor> output;
mnist::predict(input, output);

// get most likely index
Eigen::Index mostLikelyIndex = 0;
for( int output_index = 0; output_index < output.dimension(0); output_index++ )
{
    if( output[mostLikelyIndex] < output[output_index] )
    {
        mostLikelyIndex = output_index;
    }
}
...
```

## For iOS & MacOSX

we use Accelerate.framework to speed up the convolution operation.

## For Android

we only support to speed up 3x3 kernels and 1x1 stride convolution operation.

# keras layer supports

* Convolution2D
* Activation (relu, tanh, softmax)
* MaxPool
* Dense
* Flatten

# feedback

welcome post your opinion to me, you can email me as a faster way: LeiQiaTalk@hotmail.com
