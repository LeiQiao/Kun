#include <iostream>
#include "opencv2/opencv.hpp"
#include "cpp/mnist.hpp"

int main(int argc, const char* argv[])
{
    if( argc == 1 )
    {
        printf("usage: test_mnist imagefiles ...\n");
        return -1;
    }

    // read image from args
    for( int image_index = 1; image_index < argc; image_index++ )
    {
        printf("predicting file: [%s] ", argv[image_index]);

        // open image
        cv::Mat image = cv::imread(argv[image_index]);
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        if( image.data == NULL )
        {
            printf(" failed, file cannot read.\n");
            continue;
        }

        // read image data from Mat
        Eigen::Tensor<float, 3, Eigen::RowMajor> input(28, 28, 1);
        for( int row_index = 0; row_index < 28; row_index++ )
        {
            for( int col_index = 0; col_index < 28; col_index++ )
            {
                int index = col_index*28+row_index;
                input.data()[index] = image.data[index];
            }
        }
        
        // predict it
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

        printf(" \t\t\tmost likely: %ld\n", (long)mostLikelyIndex);
    }
    

    return 0;
}