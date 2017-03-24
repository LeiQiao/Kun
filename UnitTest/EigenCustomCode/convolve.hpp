//
//  convolve.hpp
//

#ifndef __CONVOLVE_HPP__
#define __CONVOLVE_HPP__

#include "Eigen.h"

#if EIGEN_OS_MAC
#include <Accelerate/Accelerate.h>
#endif

#if defined(EIGEN_VECTORIZE_NEON)
#include "conv3x3_neon.hpp"
#endif

namespace EigenCustom {


template<typename Scalar_>
void convolve(const Eigen::Tensor<Scalar_, 2, Eigen::RowMajor>& input,
              const Eigen::Tensor<Scalar_, 2, Eigen::RowMajor>& kernel,
              Eigen::Tensor<Scalar_, 2, Eigen::RowMajor>& output,
              int xStrides, int yStrides, int zeroPaddings)
{
    Eigen::Index input_dim0_length = input.dimension(0);
    Eigen::Index input_dim1_length = input.dimension(1);
    
    Eigen::Index kernel_dim0_length = kernel.dimension(0);
    Eigen::Index kernel_dim1_length = kernel.dimension(1);
    
    Eigen::Tensor<float, 2, Eigen::RowMajor> paddedInput = input;
    if( zeroPaddings > 0 )
    {
        paddedInput = Eigen::Tensor<float, 2, Eigen::RowMajor>(input_dim0_length+zeroPaddings*2,
                                                               input_dim1_length+zeroPaddings*2);
        paddedInput.setZero();
        paddedInput.slice(Eigen::DSizes<ptrdiff_t, 2>(zeroPaddings, zeroPaddings),
                          Eigen::DSizes<ptrdiff_t, 2>(input_dim0_length, input_dim1_length)) = input;
        
        input_dim0_length += zeroPaddings*2;
        input_dim1_length += zeroPaddings*2;
    }
    
#if EIGEN_OS_MAC
    if( (xStrides == 1) && (yStrides == 1) )
    {
        Eigen::Tensor<float, 2, Eigen::RowMajor> conv0_result(input_dim0_length, input_dim1_length);
        vDSP_imgfir(input.data(), input_dim0_length, input_dim1_length,
                    kernel.data(),
                    conv0_result.data(),
                    kernel_dim0_length,
                    kernel_dim1_length);
        Eigen::Index trim_dim0 = (kernel_dim0_length-1)/2;
        Eigen::Index trim_dim1 = (kernel_dim1_length-1)/2;
        output = conv0_result.slice(Eigen::DSizes<ptrdiff_t, 2>(trim_dim0, trim_dim1),
                                    Eigen::DSizes<ptrdiff_t, 2>(input_dim0_length-trim_dim0*2,
                                                                input_dim0_length-trim_dim1*2));
        return;
    }
#endif
    
#if defined(EIGEN_VECTORIZE_NEON)
    bool use_neon = ((kernel_dim0_length == 3) && (kernel_dim1_length == 3) &&
                     (xStrides == 1) && (yStrides == 1));
    Eigen::Tensor<float, 2, Eigen::RowMajor> paddedKernel = kernel;
    if( use_neon )
    {
        paddedKernel = Eigen::Tensor<float, 2, Eigen::RowMajor>(3, 4);
        paddedKernel.setZero();
        paddedKernel.slice(Eigen::DSizes<ptrdiff_t, 2>(0, 0),Eigen::DSizes<ptrdiff_t, 2>(3, 3)) = kernel;
    }
#else
    if( (xStrides == 1) && (yStrides == 1) )
    {
        Eigen::array<ptrdiff_t, 2> dims;
        dims[0] = 0;
        dims[1] = 1;
        output = input.convolve(kernel, dims);
        return;
    }
#endif // EIGEN_VECTORIZE_NEON
    
    Eigen::Index maxRows = input_dim0_length-(kernel_dim0_length-1);
    Eigen::Index maxCols = input_dim1_length-(kernel_dim1_length-1);
    for( Eigen::Index row = 0; row < maxRows; row+=yStrides )
    {
        Eigen::Index processed_cols = 0;
#if defined(EIGEN_VECTORIZE_NEON)
        if( use_neon )
        {
            processed_cols = ((input_dim0_length >> 2) << 2);
            conv3x3_neon(paddedInput.data()+(input_dim1_length*row),
                         paddedInput.data()+(input_dim1_length*(row+1)),
                         paddedInput.data()+(input_dim1_length*(row+2)),
                         paddedKernel.data(),
                         output.data()+(input_dim1_length*row),
                         processed_cols);
        }
#endif // EIGEN_VECTORIZE_NEON
        for( Eigen::Index col = processed_cols; col < maxCols; col+=xStrides )
        {
            Eigen::DSizes<ptrdiff_t, 2> slice_start(row, col);
            Eigen::DSizes<ptrdiff_t, 2> slice_length(kernel_dim0_length, kernel_dim1_length);
            
            Eigen::Tensor<float, 2, Eigen::RowMajor> inputChunk = paddedInput.slice(slice_start,
                                                                                    slice_length);
            Eigen::Tensor<float, 0, Eigen::RowMajor> value = (inputChunk * kernel).sum();
            output(row, col) = value.data()[0];
        }
    }
}


};

#endif // __CONVOLVE_HPP__
