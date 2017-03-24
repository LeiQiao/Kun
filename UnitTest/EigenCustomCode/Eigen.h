//
//  Eigen.h
//

#ifndef __EIGEN_H__
#define __EIGEN_H__


//#define EIGEN_MPL2_ONLY 1

//#define EIGEN_NO_MALLOC 1 // disable all heap allocation of matrices

#define EIGEN_USE_THREADS

#if !DEBUG

// turn off range checking, asserts, and anything else that could slow us down
// NB: turning off asserts is important! When EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO is 0,
// as it is with llvm-gcc4.2 (which is our current iOS compiler), then a very slow
// assertion macro replacement is used. As of time of writing, it cuts the 4S from
// ~22fps to ~17fps.
#define NDEBUG 1
#define EIGEN_NO_DEBUG 1

#endif

#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "EigenCustomOp.hpp"

// define RowMajor & ColMajor matrixXf
namespace Eigen {
    
typedef Eigen::MatrixXf MatrixXf_ColMajor;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_RowMajor;
    
};

#endif /* __EIGEN_H__ */
