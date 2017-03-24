//
// mnistcpp
//

#include "mnist.hpp"

#include "mnist_data.hpp"

#include "mnist_map.hpp"

namespace mnist {

#ifdef DEBUG
    
#include "sys/time.h"
double getTimeFromLastOp()
{
    static struct timeval lastTime;
    static bool haveLastTime = false;
    if( !haveLastTime )
    {
        gettimeofday(&lastTime, NULL);
        haveLastTime = true;
    }
    
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    double timeuse = curTime.tv_sec - lastTime.tv_sec + (curTime.tv_usec - lastTime.tv_usec) / 1000000.0;
    
    gettimeofday(&lastTime, NULL);
    return timeuse;
}
    
static struct timeval firstTime;

#include <sys/types.h>
#include <sys/stat.h>
#ifdef SAVE_UNIT_TEST_TENSOR
#include <stdarg.h>
#endif // SAVE_UNIT_TEST_TENSOR
void save_data_to_file(const char* filename, const uint8_t* data, unsigned long dataLen, uint32_t sectionLen, bool onByte)
{
    FILE* f = fopen(filename, "wb");
    for( int i=0; i<dataLen / sizeof(float); i++ )
    {
        char str[100];
        if( onByte )
        {
            for( int j=0; j<sizeof(float); j++ )
            {
                sprintf(str, "0x%02X, ", data[i*sizeof(float)+j]);
                fwrite(str, strlen(str), 1, f);
                if( ((i*sizeof(float)+j) % sectionLen) == (sectionLen-1) )
                {
                    sprintf(str, "\n    ");
                    fwrite(str, strlen(str), 1, f);
                }
            }
        }
        else
        {
            char fmt[20];
            if( (strcmp(typeid(float).name(), typeid(float).name()) == 0) ||
               (strcmp(typeid(float).name(), typeid(double).name()) == 0) )
            {
                sprintf(fmt, "%%.06f, ");
            }
            else
            {
                sprintf(fmt, "%%d, ");
            }
            const float* float_data = (const float*)data;
            sprintf(str, fmt, float_data[i]);
            fwrite(str, strlen(str), 1, f);
            if( (i % sectionLen) == (sectionLen-1) )
            {
                sprintf(str, "\n    ");
                fwrite(str, strlen(str), 1, f);
            }
        }
    }
    fclose(f);
}

void start_unit_test_function(const char* input_name, const float* input_tensor, int dim_count, ...)
{
    gettimeofday(&firstTime, NULL);
    
    getTimeFromLastOp();
}

void unit_test_function(const char* output_name, const float* output_tensor, int dim_count, ...)
{
    printf("%s cost: %.6f\n", output_name, getTimeFromLastOp());

#ifdef SAVE_UNIT_TEST_TENSOR
    std::vector<int> dimensions;
    int tensor_length = sizeof(float);
    
    va_list args;
    va_start(args, dim_count);
    for( int i=0; i<dim_count; i++ )
    {
        int dim_size = va_arg(args, int);
        dimensions.push_back(dim_size);
        tensor_length *= dim_size;
    }
    va_end(args);
    
    
    std::stringstream path;
    path << "./" << output_name << "/";
    rmdir(path.str().c_str());
    mkdir(path.str().c_str(), S_IRWXU|S_IRWXG|S_IRWXO);

    if( dim_count == 1 )
    {
        path << output_name;
        save_data_to_file(path.str().c_str(), (const uint8_t*)output_tensor, tensor_length, 16, false);
    }
    else if( dim_count == 3 )
    {
        Eigen::Tensor<float, 3, Eigen::RowMajor> tensor;
        tensor.resize(dimensions[0], dimensions[1], dimensions[2]);
        memcpy(tensor.data(), output_tensor, tensor_length);
        for( int i=0; i<dimensions[2]; i++ )
        {
            std::stringstream ss;
            ss << path.str() << output_name << "_" << (i+1);
            Eigen::Tensor<float, 2, Eigen::RowMajor> sub_tensor = tensor.chip<2>(i);
            save_data_to_file(ss.str().c_str(), (const uint8_t*)sub_tensor.data(), tensor_length / dimensions[2], dimensions[0], false);
        }
    }
#endif // SAVE_UNIT_TEST_TENSOR
}

void end_unit_test_function(const char* output_name, const float* output_tensor, int dim_count, ...)
{
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    double timeuse = curTime.tv_sec - firstTime.tv_sec + (curTime.tv_usec - firstTime.tv_usec) / 1000000.0;
    printf("all cost: %.6f\n", timeuse);
    printf("=========================\n");
}

#endif // DEBUG


/////////////////////////////////////////////////////////////
// conv88263 多通道卷积 数据格式 NHWC
// input: 多通道二维图片，维度<ROWS, COLS, CHANNELS>
// output: 1个卷积核卷积后的特征，维度<ROWS, COLS, CHANNELS>
void conv88263(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input, Eigen::Tensor<float, 3, Eigen::RowMajor>& output)
{
    // 将多通道转成单通道数组
    Eigen::Tensor<float, 2, Eigen::RowMajor> input_channels[1];
    for( int conv88263_channel_index = 0 ; conv88263_channel_index < 1; conv88263_channel_index++ )
    {
        input_channels[conv88263_channel_index] = input.chip<2>(conv88263_channel_index);
    }

    // 卷积核维度
    Eigen::array<ptrdiff_t, 2> conv88263_dimensions;
    conv88263_dimensions[0] = 0;
    conv88263_dimensions[1] = 1;
    
    // 卷积输出大小
    output = Eigen::Tensor<float, 3, Eigen::RowMajor>(26, 26, 32);
    
    // 所有卷积核
    Eigen::Tensor<float, 4, Eigen::RowMajor> conv88263_all_kernels(3, 3, 1, 32);
    memcpy(conv88263_all_kernels.data(), data_kernel_conv88263, 3 * 3 * 1 * 32 * sizeof(float));
    
    // 卷积核个数
    for( int conv88263_kernel_index = 0; conv88263_kernel_index < 32; conv88263_kernel_index++ )
    {
        Eigen::Tensor<float, 2, Eigen::RowMajor> conv88263_channel_result(26, 26);
        conv88263_channel_result = conv88263_channel_result.setZero();
        
        // 通道数
        for( int conv88263_channel_index = 0; conv88263_channel_index < 1; conv88263_channel_index++ )
        {
            // 获取单个卷积核
            Eigen::Tensor<float, 2, Eigen::RowMajor> conv88263_kernel(3, 3);
            conv88263_kernel = conv88263_all_kernels.chip<3>(conv88263_kernel_index).chip<2>(conv88263_channel_index);
            
            // 保存单通道的卷积结果
            Eigen::Tensor<float, 2, Eigen::RowMajor> conv88263_single_channel_result(26, 26);
            
            EigenCustom::convolve(input_channels[conv88263_channel_index],
                     conv88263_kernel,
                     conv88263_single_channel_result,
                     1, 1, 0);
            
            // 卷积计算并加上偏置
            conv88263_channel_result += conv88263_single_channel_result;
        }
        
        // 获取偏置
        float bias = 0;
        memcpy(&bias, data_bias_conv88263 + conv88263_kernel_index * sizeof(float), sizeof(float));
        
        // 保存结果
        output.chip<2>(conv88263_kernel_index) = conv88263_channel_result + bias;
    }
}

/////////////////////////////////////////////////////////////
// relu62557 激活函数
// input: 输入Tensor
// output: 输出Tensor
void relu62557(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input, Eigen::Tensor<float, 3, Eigen::RowMajor>& output)
{
    output = input.unaryExpr(EigenCustom::scalar_relu_op());
}


/////////////////////////////////////////////////////////////
// conv85557 多通道卷积 数据格式 NHWC
// input: 多通道二维图片，维度<ROWS, COLS, CHANNELS>
// output: 32个卷积核卷积后的特征，维度<ROWS, COLS, CHANNELS>
void conv85557(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input, Eigen::Tensor<float, 3, Eigen::RowMajor>& output)
{
    // 将多通道转成单通道数组
    Eigen::Tensor<float, 2, Eigen::RowMajor> input_channels[32];
    for( int conv85557_channel_index = 0 ; conv85557_channel_index < 32; conv85557_channel_index++ )
    {
        input_channels[conv85557_channel_index] = input.chip<2>(conv85557_channel_index);
    }

    // 卷积核维度
    Eigen::array<ptrdiff_t, 2> conv85557_dimensions;
    conv85557_dimensions[0] = 0;
    conv85557_dimensions[1] = 1;
    
    // 卷积输出大小
    output = Eigen::Tensor<float, 3, Eigen::RowMajor>(24, 24, 32);
    
    // 所有卷积核
    Eigen::Tensor<float, 4, Eigen::RowMajor> conv85557_all_kernels(3, 3, 32, 32);
    memcpy(conv85557_all_kernels.data(), data_kernel_conv85557, 3 * 3 * 32 * 32 * sizeof(float));
    
    // 卷积核个数
    for( int conv85557_kernel_index = 0; conv85557_kernel_index < 32; conv85557_kernel_index++ )
    {
        Eigen::Tensor<float, 2, Eigen::RowMajor> conv85557_channel_result(24, 24);
        conv85557_channel_result = conv85557_channel_result.setZero();
        
        // 通道数
        for( int conv85557_channel_index = 0; conv85557_channel_index < 32; conv85557_channel_index++ )
        {
            // 获取单个卷积核
            Eigen::Tensor<float, 2, Eigen::RowMajor> conv85557_kernel(3, 3);
            conv85557_kernel = conv85557_all_kernels.chip<3>(conv85557_kernel_index).chip<2>(conv85557_channel_index);
            
            // 保存单通道的卷积结果
            Eigen::Tensor<float, 2, Eigen::RowMajor> conv85557_single_channel_result(24, 24);
            
            EigenCustom::convolve(input_channels[conv85557_channel_index],
                     conv85557_kernel,
                     conv85557_single_channel_result,
                     1, 1, 0);
            
            // 卷积计算并加上偏置
            conv85557_channel_result += conv85557_single_channel_result;
        }
        
        // 获取偏置
        float bias = 0;
        memcpy(&bias, data_bias_conv85557 + conv85557_kernel_index * sizeof(float), sizeof(float));
        
        // 保存结果
        output.chip<2>(conv85557_kernel_index) = conv85557_channel_result + bias;
    }
}

/////////////////////////////////////////////////////////////
// relu17440 激活函数
// input: 输入Tensor
// output: 输出Tensor
void relu17440(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input, Eigen::Tensor<float, 3, Eigen::RowMajor>& output)
{
    output = input.unaryExpr(EigenCustom::scalar_relu_op());
}


/////////////////////////////////////////////////////////////
// maxpool74222 多通道池化 数据格式 NHWC
// input: 多通道二维图片，维度<ROWS, COLS, CHANNELS>
// output: 2x2池化后的特征图片，维度<ROWS, COLS, CHANNELS>
void maxpool74222(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& output)
{
#ifndef INDEX_24_24_32
#define INDEX_24_24_32(DATA, ROW, COL, CHANNEL) \
    DATA[(ROW)*(24)*(32)+(COL)*(32)+(CHANNEL)]
#endif
    
#ifndef INDEX_12_12_32
#define INDEX_12_12_32(DATA, ROW, COL, CHANNEL) \
    DATA[(ROW)*(12)*(32)+(COL)*(32)+(CHANNEL)]
#endif
    const float* srcMatrix = input.data();
    float* dstMatrix = output.data();
    
    for( int channel_index = 0; channel_index < 32; channel_index++ )
    {
        for( int row_index = 0; row_index < 12; row_index++ )
        {
            for( int col_index = 0; col_index < 12; col_index++ )
            {
                float maxValue = INDEX_24_24_32(srcMatrix, row_index*2, col_index*2, channel_index);
                for( int x = 0; x < 2; x++ )
                {
                    for( int y = 0; y < 2; y++ )
                    {
                        maxValue = Eigen::numext::maxi(maxValue, INDEX_24_24_32(srcMatrix, row_index*2+x, col_index*2+y, channel_index));
                    }
                }
                INDEX_12_12_32(dstMatrix, row_index, col_index, channel_index) = maxValue;
            }
        }
    }
}


/////////////////////////////////////////////////////////////
// flatten88662 摊平矩阵
// input: 多通道二维图片，维度<CHANNELS, ROWS, COLS>
// output: 根据HWC格式摊平成一位数组
void flatten88662(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
               Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
    // 摊平成一维
    Eigen::Tensor<float, 1>::Dimensions dims(4608);
    output = input.reshape(dims);
}


/////////////////////////////////////////////////////////////
// dense30434 矩阵相乘
// input: <4608>个元素的矩阵
// output: <128>个元素的矩阵
void dense30434(const Eigen::Tensor<float, 1, Eigen::RowMajor>& input,
              Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense30434_input_matrix(1, 4608);
    memcpy(dense30434_input_matrix.data(), input.data(), 4608*sizeof(float));
    
    // 获取权重矩阵<4608 * 128>
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense30434(4608, 128);
    memcpy(dense30434.data(), data_weight_dense30434, 4608*128*sizeof(float));
    
    Eigen::array<Eigen::Tensor<float, 1>::DimensionPair, 1> dims = {{Eigen::Tensor<float, 1>::DimensionPair(1, 0)}};
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense30434_output_matrix = dense30434_input_matrix.contract(dense30434, dims);
    
    memcpy(output.data(), dense30434_output_matrix.data(), 128*sizeof(float));

    Eigen::Tensor<float, 1, Eigen::RowMajor> bias(128);
    memcpy(bias.data(), data_bias_dense30434, 128*sizeof(float));
    output += bias;
}


/////////////////////////////////////////////////////////////
// relu27168 激活函数
// input: 输入Tensor
// output: 输出Tensor
void relu27168(const Eigen::Tensor<float, 1, Eigen::RowMajor>& input, Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
    output = input.unaryExpr(EigenCustom::scalar_relu_op());
}


/////////////////////////////////////////////////////////////
// dense66191 矩阵相乘
// input: <128>个元素的矩阵
// output: <10>个元素的矩阵
void dense66191(const Eigen::Tensor<float, 1, Eigen::RowMajor>& input,
              Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense66191_input_matrix(1, 128);
    memcpy(dense66191_input_matrix.data(), input.data(), 128*sizeof(float));
    
    // 获取权重矩阵<128 * 10>
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense66191(128, 10);
    memcpy(dense66191.data(), data_weight_dense66191, 128*10*sizeof(float));
    
    Eigen::array<Eigen::Tensor<float, 1>::DimensionPair, 1> dims = {{Eigen::Tensor<float, 1>::DimensionPair(1, 0)}};
    Eigen::Tensor<float, 2, Eigen::RowMajor> dense66191_output_matrix = dense66191_input_matrix.contract(dense66191, dims);
    
    memcpy(output.data(), dense66191_output_matrix.data(), 10*sizeof(float));

    Eigen::Tensor<float, 1, Eigen::RowMajor> bias(10);
    memcpy(bias.data(), data_bias_dense66191, 10*sizeof(float));
    output += bias;
}


/////////////////////////////////////////////////////////////
// softmax80355 线性回归
// input: <10>个元素的矩阵
// output: <10>个元素的矩阵
void softmax80355(const Eigen::Tensor<float, 1, Eigen::RowMajor>& input, Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
    float maxValue = 0.0;
    float sumValue = 0.0;
    
    for( int index = 0; index < 10; index++ )
    {
        if( maxValue < input.data()[index] )
        {
            maxValue = input.data()[index];
        }
    }
    for( int index = 0; index < 10; index++ )
    {
        output.data()[index] = exp(input.data()[index] - maxValue);
        sumValue += output.data()[index];
    }
    
    for( int index = 0; index < 10; index++ )
    {
        output.data()[index] /= sumValue;
    }
}

void predict(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
             Eigen::Tensor<float, 1, Eigen::RowMajor>& output)
{
#ifdef DEBUG
    predict(input, output, false);
}

void predict(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
             Eigen::Tensor<float, 1, Eigen::RowMajor>& output,
             bool unit_test)
{
    if( unit_test )
    {
        start_unit_test_function("input", input.data(), 3, 28, 28, 1);
    }
#endif // DEBUG
    Eigen::Tensor<float, 3, Eigen::RowMajor> output_conv88263(26, 26, 32);
    conv88263(input, output_conv88263);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("conv88263", output_conv88263.data(), 3, 26, 26, 32);
    }
#endif // DEBUG

    Eigen::Tensor<float, 3, Eigen::RowMajor> output_relu62557(26, 26, 32);
    relu62557(output_conv88263, output_relu62557);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("relu62557", output_relu62557.data(), 3, 26, 26, 32);
    }
#endif // DEBUG

    Eigen::Tensor<float, 3, Eigen::RowMajor> output_conv85557(24, 24, 32);
    conv85557(output_relu62557, output_conv85557);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("conv85557", output_conv85557.data(), 3, 24, 24, 32);
    }
#endif // DEBUG

    Eigen::Tensor<float, 3, Eigen::RowMajor> output_relu17440(24, 24, 32);
    relu17440(output_conv85557, output_relu17440);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("relu17440", output_relu17440.data(), 3, 24, 24, 32);
    }
#endif // DEBUG

    Eigen::Tensor<float, 3, Eigen::RowMajor> output_maxpool74222(12, 12, 32);
    maxpool74222(output_relu17440, output_maxpool74222);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("maxpool74222", output_maxpool74222.data(), 3, 12, 12, 32);
    }
#endif // DEBUG

    Eigen::Tensor<float, 1, Eigen::RowMajor> output_flatten88662(4608);
    flatten88662(output_maxpool74222, output_flatten88662);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("flatten88662", output_flatten88662.data(), 1, 4608);
    }
#endif // DEBUG

    Eigen::Tensor<float, 1, Eigen::RowMajor> output_dense30434(128);
    dense30434(output_flatten88662, output_dense30434);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("dense30434", output_dense30434.data(), 1, 128);
    }
#endif // DEBUG

    Eigen::Tensor<float, 1, Eigen::RowMajor> output_relu27168(128);
    relu27168(output_dense30434, output_relu27168);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("relu27168", output_relu27168.data(), 1, 128);
    }
#endif // DEBUG

    Eigen::Tensor<float, 1, Eigen::RowMajor> output_dense66191(10);
    dense66191(output_relu27168, output_dense66191);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("dense66191", output_dense66191.data(), 1, 10);
    }
#endif // DEBUG

    Eigen::Tensor<float, 1, Eigen::RowMajor> output_softmax80355(10);
    softmax80355(output_dense66191, output_softmax80355);

#ifdef DEBUG
    if( unit_test )
    {
        unit_test_function("softmax80355", output_softmax80355.data(), 1, 10);
    }
#endif // DEBUG

    output = output_softmax80355;

#ifdef DEBUG
    if( unit_test )
    {
        end_unit_test_function("output", output.data(), 1, 10);
    }
#endif // DEBUG

}
std::string getMappingTableValue(Eigen::Index index)
{
    if( index >= sizeof(value_map)/sizeof(const char*) ) return "<null>";
    else return value_map[index];
}


};

