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

float** convolve_with_gauss(float** img, int w, int h, float** gaussKernel)
{
    float** result = (float**)malloc(h * sizeof(float*));
    for (int y = 0; y < h; y++)
    {
        result[y] = (float*)malloc(w * sizeof(float));
        for (int x  = 0; x < w; x++)
        {
            int accumulator = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int tmpY = y + i - 1;
                    if (tmpY < 0)
                        tmpY = 0;
                    else if (tmpY >= h)
                        tmpY = h - 1;

                    int tmpX = x + j - 1;
                    if (tmpX < 0)
                        tmpX = 0;
                    else if (tmpX >= w)
                        tmpX = w - 1;

                    accumulator += gaussKernel[3 - i][3 - j] * img[tmpY][tmpX];
                }
            }

            result[y][x] = accumulator;
        }
    }

    return result;
}

void free_matrix(float **m, int width, int height)
{
    for (int y = 0; y < height; y++)
        free(m[y]);

    free(m);
}

void gauss_derivatives(unsigned char **img, int  width, int height, float ***dX, float ***dY)
{
    float ***derivatives = gauss_derivative_kernels();

    float **fimg = (float**)malloc(height * sizeof(float));
    for (int y = 0; y < height; y++)
    {
        fimg[y] = (float*)malloc(width * sizeof(float));
        for (int x = 0; x < width; x++)
            fimg[y][x] = (float)img[y][x];
    }

    *dX = convolve_with_gauss(fimg, width, height, derivatives[0]);
    *dY = convolve_with_gauss(fimg, width, height, derivatives[1]);

    free_matrix(fimg, width, height);
}

float** hadamarProduct(float **m1, float **m2, int width, int height)
{
    float **result = (float**)malloc(height  * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
        for (int x =  0; x < width; x++)
            result[y][x] = m1[y][x] * m2[y][x];
    }

    return result;
}

float **subtractMat(float **m1, float **m2, int width, int height)
{
    float **result = (float**)malloc(height  * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
        for (int x =  0; x < width; x++)
            result[y][x] = m1[y][x] - m2[y][x];
    }

    return result;
}

float **sumMat(float **m1, float **m2, int width, int height)
{
    float **result = (float**)malloc(height  * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
        for (int x =  0; x < width; x++)
            result[y][x] = m1[y][x] + m2[y][x];
    }

    return result;
}

float **divideMat(float **m1, float **m2, int width, int height)
{
    float **result = (float**)malloc(height  * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
        for (int x =  0; x < width; x++)
            result[y][x] = m1[y][x] / m2[y][x];
    }

    return result;
}

float **scalarSumMat(float **m1, float scalar, int width, int height)
{
    float **result = (float**)malloc(height  * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
        for (int x =  0; x < width; x++)
            result[y][x] = m1[y][x] + scalar;
    }

    return result;
}

float **compute_harris_response(unsigned char **img, int width, int height)
{
    float **imx;
    float **imy;
    gauss_derivatives(img, width, height, &imx, &imy);

    float **gauss = gauss_kernel();

    float **imx2 = hadamarProduct(imx, imx, width, height);
    float **Wxx = convolve_with_gauss(imx2, width, height, gauss);
    free_matrix(imx2, width, height);

    float **imy2 = hadamarProduct(imy, imy, width, height);
    float **Wxy = convolve_with_gauss(imy2, width, height, gauss);
    free_matrix(imy2, width, height);

    float **imxImy = hadamarProduct(imx, imy, width, height);
    float **Wyy = convolve_with_gauss(imxImy, width, height, gauss);
    free_matrix(imx, width, height);
    free_matrix(imy, width, height);
    free_matrix(gauss, width, height);
    free_matrix(imxImy, width, height);

    float **WxxWyy = hadamarProduct(Wxx, Wyy, width, height);
    float **Wxy2 = hadamarProduct(Wxy, Wxy, width, height);
    float **Wdet = subtractMat(WxxWyy, Wxy2, width, height);
    free_matrix(WxxWyy, width, height);
    free_matrix(Wxy2, width, height);
    
    float **Wtr = sumMat(Wxx, Wyy, width, height);
    free_matrix(Wxx, width, height);
    free_matrix(Wyy, width, height);
    
    float **WtrEps = scalarSumMat(Wtr, 1, width, height);
    free_matrix(Wtr, width, height);

    float **result = divideMat(Wdet, WtrEps, width, height);
    free_matrix(Wdet, width, height);
    free_matrix(WtrEps, width, height);

    return result;
}