#include <iostream>
#include "unit_test_data.hpp"
#include "%MODEL_NAME%.hpp"

int main()
{
    Eigen::Tensor<%CPP_TYPE%, 3, Eigen::RowMajor> input(%DIM_SIZE_1%, %DIM_SIZE_2%, %DIM_SIZE_3%);
    memcpy(input.data(), uint_test_data::unit_test_data, %DIM_SIZE_1%*%DIM_SIZE_2%*%DIM_SIZE_3%*sizeof(%CPP_TYPE%));
    
    Eigen::Tensor<%CPP_TYPE%, 1, Eigen::RowMajor> output;
    %MODEL_NAME%::predict(input, output, true);

    FILE* file = fopen("./unit_test_input.txt", "wb");
    for( int i=0; i<%DIM_SIZE_1%*%DIM_SIZE_2%*%DIM_SIZE_3%; i++ )
    {
        char str[100];
        sprintf(str, "%.06f,", input.data()[i]);
        fwrite(str, strlen(str), 1, file);
    }

    return 0;
}