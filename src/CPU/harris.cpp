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

bool **getEllipse()
{
    bool staticEllipse[25][25] = {
        {false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false},
       {false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false},
       {false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false},
       {false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false},
       {false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false},
       {false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false},
       {false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false},
       {false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false},
       {false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true},
       {false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false},
       {false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false},
       {false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false},
       {false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false},
       {false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false},
       {false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false},
       {false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false},
       {false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false},
       {false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false}
    };

    bool** ellipse = (bool**)malloc(7 * sizeof(bool*));
    for (int i = 0; i < 7; i++)
    {
        ellipse[i] = (bool*)malloc(7 * sizeof(bool));
        memcpy(ellipse[i], staticEllipse[i], 7 * sizeof(bool));
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

unsigned char **morpho_erode(unsigned char **img, int width, int height, int threshold)
{
    bool **structElement = getEllipse();
    unsigned char **result = (unsigned char**)malloc(height * sizeof(unsigned char*));
    
    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)malloc(width * sizeof(unsigned char));
        
        int imin = min(0, y - 12);
        int imax = max(y + 12, height - 1);
        for (int x = 0; x < height; x++)
        {
            int jmin = min(0, x - 12);
            int jmax = max(x + 12, width - 1);
            
            unsigned char pixel = img[y][x];
            if (pixel == 0)
                continue;

            for (int i = imin; i < imax + 1; i++)
            {
                for (int j = jmin; j < jmax + 1; j++)
                {
                    if (structElement[i - imin][j - jmin] && img[i][j] <= threshold)
                    {
                        pixel = 0;
                        break;
                    }
                }
                if (pixel == 0)
                    break;
            }

            result[y][x] = pixel;
        }
    }
}

unsigned char **morpho_dilate(unsigned char **img, int width, int height, int threshold)
{
    bool **structElement = getEllipse();
    unsigned char **result = (unsigned char**)malloc(height * sizeof(unsigned char*));

    for (int y = 0; y < height; y++)
    {
        result[y] = (unsigned char*)malloc(width * sizeof(unsigned char));
        
        int imin = min(0, y - 12);
        int imax = max(y + 12, height - 1);
        
        for (int x = 0; x < width; x++)
        {
            int jmin = min(0, x - 12);
            int jmax = max(x + 12, width - 1);
            
            unsigned char pixel = 0;
            if (pixel > 0)
                continue;

            for (int i = imin; i < imax + 1; i++)
            {
                for (int j = jmin; j < jmax + 1; j++)
                {
                    if (structElement[i - imin][j - jmin] && img[i][j] > threshold)
                    {
                        pixel = img[i][j];
                        break;
                    }
                }
                if (pixel > 0)
                    break;
            }

            result[y][x] = pixel;
        }
    }
}

void free_image(unsigned char **img, int h)
{
    for (int i = 0; i < h; i++)
        free(img[i]);

    free(img);
}

unsigned char** morpho_open(unsigned char **img, int width, int height, int threshold)
{
    unsigned char **eroded = morpho_erode(img, width, height, threshold);
    unsigned char **opened = morpho_dilate(eroded, width, height, threshold);

    free_image(eroded, height);

    return opened;
}