#include "harris.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

typedef unsigned char ** png_bytepp;

// Debug
#pragma region
void printMinMax(float **m, int width, int height)
{
    float minVal = m[0][0];
    float maxVal = m[0][0];
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float val = m[y][x];
            if (val < minVal)
                minVal = val;
            else if (val > maxVal)
                maxVal = val;
        }
    }

    std::cout << "Min:" << minVal << " Max:" << maxVal << "\n";
}
#pragma endregion

// Kernels
#pragma region
float** gauss_kernel()
{
    float staticKernel[7][7] = {
        // sigma = size/3
        // {1.23409804e-04,1.50343919e-03,6.73794700e-03,1.11089965e-02,6.73794700e-03,1.50343919e-03,1.23409804e-04},
        // {1.50343919e-03,1.83156389e-02,8.20849986e-02,1.35335283e-01,8.20849986e-02,1.83156389e-02,1.50343919e-03},
        // {6.73794700e-03,8.20849986e-02,3.67879441e-01,6.06530660e-01,3.67879441e-01,8.20849986e-02,6.73794700e-03},
        // {1.11089965e-02,1.35335283e-01,6.06530660e-01,1.00000000e+00,6.06530660e-01,1.35335283e-01,1.11089965e-02},
        // {6.73794700e-03,8.20849986e-02,3.67879441e-01,6.06530660e-01,3.67879441e-01,8.20849986e-02,6.73794700e-03},
        // {1.50343919e-03,1.83156389e-02,8.20849986e-02,1.35335283e-01,8.20849986e-02,1.83156389e-02,1.50343919e-03},
        // {1.23409804e-04,1.50343919e-03,6.73794700e-03,1.11089965e-02,6.73794700e-03,1.50343919e-03,1.23409804e-04}

        // sigma = 0.33 * size
        {1.02798843e-04,1.31755659e-03,6.08748501e-03,1.01389764e-02,6.08748501e-03,1.31755659e-03,1.02798843e-04},
        {1.31755659e-03,1.68869153e-02,7.80223366e-02,1.29949664e-01,7.80223366e-02,1.68869153e-02,1.31755659e-03},
        {6.08748501e-03,7.80223366e-02,3.60485318e-01,6.00404295e-01,3.60485318e-01,7.80223366e-02,6.08748501e-03},
        {1.01389764e-02,1.29949664e-01,6.00404295e-01,1.00000000e+00,6.00404295e-01,1.29949664e-01,1.01389764e-02},
        {6.08748501e-03,7.80223366e-02,3.60485318e-01,6.00404295e-01,3.60485318e-01,7.80223366e-02,6.08748501e-03},
        {1.31755659e-03,1.68869153e-02,7.80223366e-02,1.29949664e-01,7.80223366e-02,1.68869153e-02,1.31755659e-03},
        {1.02798843e-04,1.31755659e-03,6.08748501e-03,1.01389764e-02,6.08748501e-03,1.31755659e-03,1.02798843e-04}
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
            // sigma = size/3
            // {3.70229412e-04,3.00687839e-03,6.73794700e-03,0.00000000e+00,-6.73794700e-03,-3.00687839e-03,-3.70229412e-04},
            // {4.51031758e-03,3.66312778e-02,8.20849986e-02,0.00000000e+00,-8.20849986e-02,-3.66312778e-02,-4.51031758e-03},
            // {2.02138410e-02,1.64169997e-01,3.67879441e-01,0.00000000e+00,-3.67879441e-01,-1.64169997e-01,-2.02138410e-02},
            // {3.33269896e-02,2.70670566e-01,6.06530660e-01,0.00000000e+00,-6.06530660e-01,-2.70670566e-01,-3.33269896e-02},
            // {2.02138410e-02,1.64169997e-01,3.67879441e-01,0.00000000e+00,-3.67879441e-01,-1.64169997e-01,-2.02138410e-02},
            // {4.51031758e-03,3.66312778e-02,8.20849986e-02,0.00000000e+00,-8.20849986e-02,-3.66312778e-02,-4.51031758e-03},
            // {3.70229412e-04,3.00687839e-03,6.73794700e-03,0.00000000e+00,-6.73794700e-03,-3.00687839e-03,-3.70229412e-04}

            // sigma = 0.33 * size
            {3.08396530e-04,2.63511317e-03,6.08748501e-03,0.00000000e+00,-6.08748501e-03,-2.63511317e-03,-3.08396530e-04},
            {3.95266976e-03,3.37738305e-02,7.80223366e-02,0.00000000e+00,-7.80223366e-02,-3.37738305e-02,-3.95266976e-03},
            {1.82624550e-02,1.56044673e-01,3.60485318e-01,0.00000000e+00,-3.60485318e-01,-1.56044673e-01,-1.82624550e-02},
            {3.04169293e-02,2.59899329e-01,6.00404295e-01,0.00000000e+00,-6.00404295e-01,-2.59899329e-01,-3.04169293e-02},
            {1.82624550e-02,1.56044673e-01,3.60485318e-01,0.00000000e+00,-3.60485318e-01,-1.56044673e-01,-1.82624550e-02},
            {3.95266976e-03,3.37738305e-02,7.80223366e-02,0.00000000e+00,-7.80223366e-02,-3.37738305e-02,-3.95266976e-03},
            {3.08396530e-04,2.63511317e-03,6.08748501e-03,0.00000000e+00,-6.08748501e-03,-2.63511317e-03,-3.08396530e-04}
        },
        {
            // sigma = size/3
            // {3.70229412e-04,4.51031758e-03,2.02138410e-02,3.33269896e-02,2.02138410e-02,4.51031758e-03,3.70229412e-04},
            // {3.00687839e-03,3.66312778e-02,1.64169997e-01,2.70670566e-01,1.64169997e-01,3.66312778e-02,3.00687839e-03},
            // {6.73794700e-03,8.20849986e-02,3.67879441e-01,6.06530660e-01,3.67879441e-01,8.20849986e-02,6.73794700e-03},
            // {0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00},
            // {-6.73794700e-03,-8.20849986e-02,-3.67879441e-01,-6.06530660e-01,-3.67879441e-01,-8.20849986e-02,-6.73794700e-03},
            // {-3.00687839e-03,-3.66312778e-02,-1.64169997e-01,-2.70670566e-01,-1.64169997e-01,-3.66312778e-02,-3.00687839e-03},
            // {-3.70229412e-04,-4.51031758e-03,-2.02138410e-02,-3.33269896e-02,-2.02138410e-02,-4.51031758e-03,-3.70229412e-04}

            // sigma = 0.33 * size
            {3.08396530e-04,3.95266976e-03,1.82624550e-02,3.04169293e-02,1.82624550e-02,3.95266976e-03,3.08396530e-04},
            {2.63511317e-03,3.37738305e-02,1.56044673e-01,2.59899329e-01,1.56044673e-01,3.37738305e-02,2.63511317e-03},
            {6.08748501e-03,7.80223366e-02,3.60485318e-01,6.00404295e-01,3.60485318e-01,7.80223366e-02,6.08748501e-03},
            {0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00},
            {-6.08748501e-03,-7.80223366e-02,-3.60485318e-01,-6.00404295e-01,-3.60485318e-01,-7.80223366e-02,-6.08748501e-03},
            {-2.63511317e-03,-3.37738305e-02,-1.56044673e-01,-2.59899329e-01,-1.56044673e-01,-3.37738305e-02,-2.63511317e-03},
            {-3.08396530e-04,-3.95266976e-03,-1.82624550e-02,-3.04169293e-02,-1.82624550e-02,-3.95266976e-03,-3.08396530e-04}
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
#pragma endregion

// Matrix utilities
#pragma region
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

void free_matrix(float **m, int height)
{
    for (int y = 0; y < height; y++)
        free(m[y]);

    free(m);
}
#pragma endregion

// Utilities
#pragma region
png_bytepp getEllipse()
{
    unsigned char staticEllipse[25][25] = {
        {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0},
        {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0},
        {0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
        {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}
    };

    unsigned char** ellipse = (unsigned char**)malloc(25 * sizeof(unsigned char*));
    for (int i = 0; i < 25; i++)
    {
        ellipse[i] = (unsigned char*)malloc(25 * sizeof(unsigned char));
        memcpy(ellipse[i], staticEllipse[i], 25 * sizeof(unsigned char));
    }

    return ellipse;
}

int min(int a, int b)
{
    return a < b ? a : b;
}

int max(int a, int b)
{
    return a > b ? a : b;
}

bool isClose(float a, float b)
{
    float delta = a - b;
    if (delta < 0)
        delta = -delta;
        
    return delta <= __FLT_EPSILON__;
}

void freeImage(png_bytepp img, int h)
{
    for (int i = 0; i < h; i++)
        free(img[i]);

    free(img);
}
#pragma endregion

float** convolve_with_gauss(float** img, int w, int h, float** gaussKernel)
{
    float** result = (float**)malloc(h * sizeof(float*));
    for (int y = 0; y < h; y++)
    {
        result[y] = (float*)malloc(w * sizeof(float));
        for (int x  = 0; x < w; x++)
        {
            float accumulator = 0;
            for (int i = 0; i < 7; i++)
            {
                for (int j = 0; j < 7; j++)
                {
                    int tmpY = y + i - 3;
                    if (tmpY < 0)
                        tmpY = 0;
                    else if (tmpY >= h)
                        tmpY = h - 1;

                    int tmpX = x + j - 3;
                    if (tmpX < 0)
                        tmpX = 0;
                    else if (tmpX >= w)
                        tmpX = w - 1;

                    accumulator += gaussKernel[6-i][6-j] * img[tmpY][tmpX];
                }
            }

            result[y][x] = accumulator;
        }
    }

    return result;
}

void gauss_derivatives(png_bytepp img, int  width, int height, float ***dX, float ***dY)
{
    float ***derivatives = gauss_derivative_kernels();

    float **fimg = (float**)malloc(height * sizeof(float*));
    for (int y = 0; y < height; y++)
    {
        fimg[y] = (float*)malloc(width * sizeof(float));
        for (int x = 0; x < width; x++)
            fimg[y][x] = (float)img[y][x];
    }

    *dX = convolve_with_gauss(fimg, width, height, derivatives[0]);
    *dY = convolve_with_gauss(fimg, width, height, derivatives[1]);

    free_matrix(fimg, height);
    free_matrix(derivatives[0], 7);
    free_matrix(derivatives[1], 7);
    free(derivatives);
}

float **computeHarrisResponse(png_bytepp img, int width, int height)
{
    float **imx;
    float **imy;
    gauss_derivatives(img, width, height, &imx, &imy);

    float **gauss = gauss_kernel();

    float **imx2 = hadamarProduct(imx, imx, width, height);
    float **Wxx = convolve_with_gauss(imx2, width, height, gauss);
    free_matrix(imx2, height);
    printMinMax(Wxx, width, height);

    float **imxImy = hadamarProduct(imx, imy, width, height);
    float **Wxy = convolve_with_gauss(imxImy, width, height, gauss);
    free_matrix(imxImy, height);
    printMinMax(Wxy, width, height);

    float **imy2 = hadamarProduct(imy, imy, width, height);
    float **Wyy = convolve_with_gauss(imy2, width, height, gauss);
    free_matrix(imx, height);
    free_matrix(imy, height);
    free_matrix(imy2, height);
    printMinMax(Wyy, width, height);

    float **WxxWyy = hadamarProduct(Wxx, Wyy, width, height);
    float **Wxy2 = hadamarProduct(Wxy, Wxy, width, height);
    free_matrix(Wxy, height);
    float **Wdet = subtractMat(WxxWyy, Wxy2, width, height);
    free_matrix(WxxWyy, height);
    free_matrix(Wxy2, height);
    
    float **Wtr = sumMat(Wxx, Wyy, width, height);
    free_matrix(Wxx, height);
    free_matrix(Wyy, height);
    
    float **WtrEps = scalarSumMat(Wtr, 1, width, height);
    free_matrix(Wtr, height);

    float **result = divideMat(Wdet, WtrEps, width, height);
    free_matrix(Wdet, height);
    free_matrix(WtrEps, height);

    return result;
}

png_bytepp morphoErode(png_bytepp img, int width, int height)
{
    png_bytepp structElement = getEllipse();
    png_bytepp result = (unsigned char**)calloc(height, sizeof(unsigned char*));

    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)calloc(width, sizeof(unsigned char));
        if (y < 12 || y >= height - 12)
            continue;
            
        for (int x = 12; x < width - 12; x++)
        {   
            unsigned char pixel = img[y][x];

            for (int i = y - 12; pixel != 0 && i <= y + 12; i++)
            {
                for (int j = x - 12 ; pixel != 0 && j <= x + 12; j++)
                {
                    if (structElement[i - y + 12][j - x + 12] && img[i][j] == 0)
                        pixel = 0;
                }
            }

            result[y][x] = pixel;
        }
    }

    freeImage(structElement, 25);

    return result;
}

float **morphoDilate(float **img, int width, int height)
{
    png_bytepp structElement = getEllipse();
    float **result = (float**)malloc(height * sizeof(float*));

    for (int y = 0; y < height; y++)
    {
        result[y] = (float*)malloc(width * sizeof(float));
            
        for (int x = 0; x < width; x++)
        {   
            float pixel = 0;

            for (int i = 0; i < 25; i++)
            {
                int testedY = y + i - 12;
                for (int j = 0; j < 25; j++)
                {
                    int testedX = x + j - 12;
                    if (testedY >= 0 && testedY < height
                        && testedX >= 0 && testedX < width
                        && structElement[i][j] 
                        && img[testedY][testedX] > pixel)
                        pixel = img[testedY][testedX];
                }
            }

            result[y][x] = pixel;
        }
    }

    freeImage(structElement, 25);

    return result;
}

unsigned char** harrisThreshold(float **harris, int width, int height, float threshold)
{
    png_bytepp result = (unsigned char**)malloc(height * sizeof(unsigned char*));
    
    float minVal = harris[0][0];
    float maxVal = harris[0][0];
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float val = harris[y][x];
            if (val < minVal)
                minVal = val;

            if (val > maxVal)
                maxVal = val;
        }
    }
    
    float ref = minVal + threshold * (maxVal - minVal);
    //std::cout << "Ref:" << ref << " Threshold:" << threshold << " Max:" << maxVal << " Min:" << minVal << "\n";
    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)malloc(width * sizeof(unsigned char));
        for (int x = 0; x < width; x++)
            result[y][x] = harris[y][x] > ref ? 255 : 0;
    }

    return result;
}

std::vector<std::tuple<float, int, int>> detectHarrisPoints(png_bytepp image, int width, int height, size_t max_keypoints, float threshold)
{
    float **harrisResponse = computeHarrisResponse(image, width, height);

    png_bytepp erodedMask = morphoErode(image, width, height);
    png_bytepp harrisThresholdMask = harrisThreshold(harrisResponse, width, height, threshold);
    float **dilatedMask = morphoDilate(harrisResponse, width, height);

    std::vector<std::tuple<float, int, int>> keypoints = std::vector<std::tuple<float, int, int>>();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (erodedMask[y][x] > 0 && harrisThresholdMask[y][x] > 0 && isClose(dilatedMask[y][x], harrisResponse[y][x]))
                keypoints.push_back(std::tuple<float, int, int>(harrisResponse[y][x], y, x));
        }
    }

    free_matrix(harrisResponse, height);
    freeImage(erodedMask, height);
    freeImage(harrisThresholdMask, height);
    free_matrix(dilatedMask, height);

    std::sort(keypoints.begin(), keypoints.end(), [](std::tuple<float, int, int> a, std::tuple<float, int, int> b){return std::get<0>(a)>std::get<0>(b);});

    std::vector<std::tuple<float, int, int>> limitedKeypoints = std::vector<std::tuple<float, int, int>>();
    for (size_t i = 0; i < max_keypoints && i < keypoints.size(); i++)
        limitedKeypoints.push_back(keypoints[i]);

    return keypoints;
}