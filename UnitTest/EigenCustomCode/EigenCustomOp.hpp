//
//  EigenCustomOp.hpp
//

#ifndef __EIGEN_CUSTOM_OP_HPP__
#define __EIGEN_CUSTOM_OP_HPP__

#include "convolve.hpp"

namespace EigenCustom {
/**
 *
 * \brief Template functor to compute the relu of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::exp()
 */
struct scalar_relu_op {
    template<typename Scalar>
    EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const {static Scalar zero=0; return Eigen::numext::maxi(a, zero); }
};

};

#endif /* __EIGEN_CUSTOM_OP_HPP__ */
