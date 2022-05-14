#include "harris.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

typedef unsigned char ** png_bytepp;

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
            for (int i = 0; i < 7; i++)
            {
                for (int j = 0; j < 7; j++)
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

                    accumulator += gaussKernel[6 - i][6 - j] * img[tmpY][tmpX];
                }
            }

            result[y][x] = accumulator;
        }
    }

    return result;
}

void free_matrix(float **m, int height)
{
    for (int y = 0; y < height; y++)
        free(m[y]);

    free(m);
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

float **computeHarrisResponse(png_bytepp img, int width, int height)
{
    float **imx;
    float **imy;
    gauss_derivatives(img, width, height, &imx, &imy);

    float **gauss = gauss_kernel();

    float **imx2 = hadamarProduct(imx, imx, width, height);
    float **Wxx = convolve_with_gauss(imx2, width, height, gauss);
    free_matrix(imx2, height);

    float **imy2 = hadamarProduct(imy, imy, width, height);
    float **Wxy = convolve_with_gauss(imy2, width, height, gauss);
    free_matrix(imy2, height);

    float **imxImy = hadamarProduct(imx, imy, width, height);
    float **Wyy = convolve_with_gauss(imxImy, width, height, gauss);
    free_matrix(imx, height);
    free_matrix(imy, height);
    free_matrix(gauss, 7);
    free_matrix(imxImy, height);

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

void freeImage(png_bytepp img, int h)
{
    for (int i = 0; i < h; i++)
        free(img[i]);

    free(img);
}

png_bytepp morphoErode(png_bytepp img, int width, int height)
{
    png_bytepp structElement = getEllipse();
    png_bytepp result = (unsigned char**)malloc(height * sizeof(unsigned char*));
    
    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)malloc(width * sizeof(unsigned char));
        
        int imin = max(0, y - 12);
        int imax = min(y + 12, height - 1);
        for (int x = 0; x < height; x++)
        {
            int jmin = max(0, x - 12);
            int jmax = min(x + 12, width - 1);
            
            unsigned char pixel = img[y][x];

            for (int i = imin; i < imax + 1; i++)
            {
                for (int j = jmin; j < jmax + 1; j++)
                {
                    if (structElement[i - imin][j - jmin] && img[i][j] < pixel)
                        pixel = img[i][j];
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
        
        int imin = max(0, y - 12);
        int imax = min(y + 12, height - 1);
        
        for (int x = 0; x < width; x++)
        {
            int jmin = max(0, x - 12);
            int jmax = min(x + 12, width - 1);
            
            float pixel = 0;

            for (int i = imin; i < imax + 1; i++)
            {
                for (int j = jmin; j < jmax + 1; j++)
                {
                    if (structElement[i - imin][j - jmin] && img[i][j] > pixel)
                        pixel = img[i][j];
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
    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)malloc(width * sizeof(unsigned char));
        for (int x = 0; x < width; x++)
            result[y][x] = harris[y][x] > ref ? 1 : 0;
    }

    return result;
}

bool isClose(float a, float b)
{
    float delta = a - b;
    if (delta < 0)
        delta = -delta;
        
    return delta == __FLT_EPSILON__;
}

std::vector<std::tuple<float, int, int>> detectHarrisPoints(png_bytepp image, int width, int height, int max_keypoints, float threshold)
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
    for (int i = 0; i < max_keypoints && i < keypoints.size(); i++)
        limitedKeypoints.push_back(keypoints[i]);

    return keypoints;
}