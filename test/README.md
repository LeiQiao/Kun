# MNIST
we had trained a simple CNN model in "mnist_keras_model" folder, and dumped to cpp files in "cpp" folder

## main.cpp
this file show how to predict in your own project.

## eigen
this folder is a symbol linker, you should download eigen and replace it.

## OpenCV
our main.cpp use OpenCV to load images, we recommended you download opencv v3.2.0

## cmake
```
mkdir build
cd build
cmake ..
make
mnist_predict ../0.png ../1.png
```


