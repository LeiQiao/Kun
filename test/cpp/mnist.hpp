//
//  mnist.hpp
//

#ifndef __MNIST_HPP__
#define __MNIST_HPP__

#include "Eigen.h"

namespace mnist {

void predict(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
             Eigen::Tensor<float, 1, Eigen::RowMajor>& output);

#ifdef DEBUG
void predict(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
             Eigen::Tensor<float, 1, Eigen::RowMajor>& output,
             bool unit_test);
#endif

std::string getMappingTableValue(Eigen::Index index);

};

#endif /* __MNIST_HPP__ */
