#include "harris.hpp"

#include <cstdlib>
#include <cstring>

float** gauss_kernel()
{
    float staticKernel[7][7] = {
        {0.0000123, 0.0005034, 0.0006738, 0.0011109, 0.0006738, 0.0001503, 0.0000123},
        {0.0001503, 0.0018316, 0.0082085, 0.0135335, 0.0082085, 0.0018316, 0.0001503},
        {0.0006738, 0.0082085, 0.0367879, 0.0606531, 0.0367879, 0.0082085, 0.0006738},
        {0.0011109, 0.0135335, 0.0606531, 1.0000000, 0.0606531, 0.0135335, 0.0011109},
        {0.0006738, 0.0082085, 0.0367879, 0.0606531, 0.0367879, 0.0082085, 0.0006738},
        {0.0001503, 0.0018316, 0.0082085, 0.0135335, 0.0082085, 0.0018316, 0.0001503},
        {0.0000123, 0.0005034, 0.0006738, 0.0011109, 0.0006738, 0.0001503, 0.0000123}
    };

    float** kernel = (float**)malloc(7 * sizeof(float*));
    for (int i = 0; i < 7; i++)
    {
        kernel[i] = (float*)malloc(7 * sizeof(float));
        memcpy(kernel[i], staticKernel[i], 7 * sizeof(float));
    }

    return kernel;
}

float*** gauss_derivative_kernels()
{
    float staticKernels[2][7][7] = {
        {
            {0.0003702, 0.0030069, 0.0067379, 0.0000000, -0.0067379, -0.0030069, -0.0003702},
            {0.0045103, 0.0366313, 0.0820850, 0.0000000, -0.0820850, -0.0366313, -0.0045103},
            {0.0202138, 0.1641700, 0.3678794, 0.0000000, -0.3678794, -0.1641700, -0.0202138},
            {0.0333270, 0.2706706, 0.6065307, 0.0000000, -0.6065307, -0.2706706, -0.0333270},
            {0.0202138, 0.1641700, 0.3678794, 0.0000000, -0.3678794, -0.1641700, -0.0202138},
            {0.0045103, 0.0366313, 0.0820850, 0.0000000, -0.0820850, -0.0366313, -0.0045103},
            {0.0003702, 0.0030069, 0.0067379, 0.0000000, -0.0067379, -0.0030069, -0.0003702},
        },
        {
            {0.0003702, 0.0045103, 0.0202138, 0.0333270, 0.0202138, 0.0045103, 0.0003703},
            {0.0030069, 0.0366313, 0.1641700, 0.2706706, 0.1641700, 0.0366313, 0.0030069},
            {0.0067379, 0.0820850, 0.3678794, 0.6065307, 0.3678794, 0.0820850, 0.0067379},
            {0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000},
            {-0.0067379, -0.0820850, -0.3678794, -0.6065307, -0.3678794, -0.0820850, -0.0067379},
            {-0.0030069, -0.0366313, -0.1641700, -0.2706706, -0.1641700, -0.0366313, -0.0030069},
            {-0.0003702, -0.0045103, -0.0202138, -0.0333270, -0.0202138, -0.0045103, -0.0003703},
        }
    };

    float ***kernels = (float***)malloc(2 * sizeof(float**));
    for (int i = 0; i < 2; i++)
    {
        kernels[i] = (float**)malloc(7 * sizeof(float*));
        for (int j = 0; j < 7; j++)
        {
            kernels[i][j] = (float*)malloc(7 * sizeof(float));
            memcpy(kernels[i][j], staticKernels[i][j], 7 * sizeof(float));
        }
    }

    return kernels;
}