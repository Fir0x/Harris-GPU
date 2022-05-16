#include <png.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cassert>
#include <cstdio>

#include "harris.hpp"

#define RGBA_DIM 4

[[gnu::noinline]] void _abortError(const char *msg, const char *fname,
                                   int line) {
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__constant__ float gauss_kernel[7][7] = {
    {1.23409804e-04, 1.50343919e-03, 6.73794700e-03, 1.11089965e-02,
     6.73794700e-03, 1.50343919e-03, 1.23409804e-04},
    {1.50343919e-03, 1.83156389e-02, 8.20849986e-02, 1.35335283e-01,
     8.20849986e-02, 1.83156389e-02, 1.50343919e-03},
    {6.73794700e-03, 8.20849986e-02, 3.67879441e-01, 6.06530660e-01,
     3.67879441e-01, 8.20849986e-02, 6.73794700e-03},
    {1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.00000000e+00,
     6.06530660e-01, 1.35335283e-01, 1.11089965e-02},
    {6.73794700e-03, 8.20849986e-02, 3.67879441e-01, 6.06530660e-01,
     3.67879441e-01, 8.20849986e-02, 6.73794700e-03},
    {1.50343919e-03, 1.83156389e-02, 8.20849986e-02, 1.35335283e-01,
     8.20849986e-02, 1.83156389e-02, 1.50343919e-03},
    {1.23409804e-04, 1.50343919e-03, 6.73794700e-03, 1.11089965e-02,
     6.73794700e-03, 1.50343919e-03, 1.23409804e-04}};

__constant__ float gauss_derivative_x[7][7] = {
    {3.70229412e-04, 3.00687839e-03, 6.73794700e-03, 0.00000000e+00,
     -6.73794700e-03, -3.00687839e-03, -3.70229412e-04},
    {4.51031758e-03, 3.66312778e-02, 8.20849986e-02, 0.00000000e+00,
     -8.20849986e-02, -3.66312778e-02, -4.51031758e-03},
    {2.02138410e-02, 1.64169997e-01, 3.67879441e-01, 0.00000000e+00,
     -3.67879441e-01, -1.64169997e-01, -2.02138410e-02},
    {3.33269896e-02, 2.70670566e-01, 6.06530660e-01, 0.00000000e+00,
     -6.06530660e-01, -2.70670566e-01, -3.33269896e-02},
    {2.02138410e-02, 1.64169997e-01, 3.67879441e-01, 0.00000000e+00,
     -3.67879441e-01, -1.64169997e-01, -2.02138410e-02},
    {4.51031758e-03, 3.66312778e-02, 8.20849986e-02, 0.00000000e+00,
     -8.20849986e-02, -3.66312778e-02, -4.51031758e-03},
    {3.70229412e-04, 3.00687839e-03, 6.73794700e-03, 0.00000000e+00,
     -6.73794700e-03, -3.00687839e-03, -3.70229412e-04}};

__constant__ float gauss_derivative_y[7][7] = {
    {3.70229412e-04, 4.51031758e-03, 2.02138410e-02, 3.33269896e-02,
     2.02138410e-02, 4.51031758e-03, 3.70229412e-04},
    {3.00687839e-03, 3.66312778e-02, 1.64169997e-01, 2.70670566e-01,
     1.64169997e-01, 3.66312778e-02, 3.00687839e-03},
    {6.73794700e-03, 8.20849986e-02, 3.67879441e-01, 6.06530660e-01,
     3.67879441e-01, 8.20849986e-02, 6.73794700e-03},
    {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00},
    {-6.73794700e-03, -8.20849986e-02, -3.67879441e-01, -6.06530660e-01,
     -3.67879441e-01, -8.20849986e-02, -6.73794700e-03},
    {-3.00687839e-03, -3.66312778e-02, -1.64169997e-01, -2.70670566e-01,
     -1.64169997e-01, -3.66312778e-02, -3.00687839e-03},
    {-3.70229412e-04, -4.51031758e-03, -2.02138410e-02, -3.33269896e-02,
     -2.02138410e-02, -4.51031758e-03, -3.70229412e-04}};

__constant__ unsigned char structElement[25][25] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

// Device code
__global__ void rgba2gray(unsigned char *rgba_buffer, float *gray_buffer,
                          int width, int height, size_t rgba_pitch,
                          size_t gray_pitch) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int y_rgba = y * rgba_pitch;
  gray_buffer[y * gray_pitch + x] = 0.299 * rgba_buffer[y_rgba + x * 4] +
                                    0.587 * rgba_buffer[y_rgba + x * 4 + 1] +
                                    0.114 * rgba_buffer[y_rgba + x * 4 + 2];
}

__device__ float convolve(float *matrix, int width, int height, size_t pitch,
                          float kernel[7][7]) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float accumulator = 0;

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      int tmpY = min(max(0, y + i - 3), height - 1);
      int tmpX = min(max(0, x + j - 3), width - 1);

      accumulator += kernel[6 - i][6 - j] * matrix[tmpY * pitch + tmpX];
    }
  }

  return accumulator;
}

__global__ void computeDerivatives(float *image, int width, int height,
                                   size_t pitch, float *imx2, float *imxy,
                                   float *imy2) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float imx = 0;
  float imy = 0;

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      int tmpY = min(max(0, y + i - 3), height - 1);
      int tmpX = min(max(0, x + j - 3), width - 1);

      imx += gauss_derivative_x[6 - i][6 - j] * image[tmpY * pitch + tmpX];
      imy += gauss_derivative_y[6 - i][6 - j] * image[tmpY * pitch + tmpX];
    }
  }

  imx2[y * pitch + x] = imx * imx;
  imxy[y * pitch + x] = imx * imy;
  imy2[y * pitch + x] = imy * imy;
}

__global__ void computeHarrisResponse(int width, int height, size_t pitch,
                                      float *imx2, float *imxy, float *imy2,
                                      float *response) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float Wxx = 0;
  float Wxy = 0;
  float Wyy = 0;

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      int tmpY = min(max(0, y + i - 3), height - 1);
      int tmpX = min(max(0, x + j - 3), width - 1);

      Wxx += gauss_kernel[6 - i][6 - j] * imx2[tmpY * pitch + tmpX];
      Wxy += gauss_kernel[6 - i][6 - j] * imxy[tmpY * pitch + tmpX];
      Wyy += gauss_kernel[6 - i][6 - j] * imy2[tmpY * pitch + tmpX];
    }
  }

  float WxxWyy = Wxx * Wyy;
  float Wxy2 = Wxy * Wxy;

  float Wdet = WxxWyy - Wxy2;
  float WtrEps = Wxx + Wyy + 1;

  response[y * pitch + x] = Wdet / WtrEps;
}

__global__ void morphoDilate(float *input, int width, int height, size_t pitch,
                             float *output) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float pixel = 0;

  for (int i = 0; i < 25; i++) {
    int testedY = y + i - 12;
    for (int j = 0; j < 25; j++) {
      int testedX = x + j - 12;
      if (testedY >= 0 && testedY < height && testedX >= 0 && testedX < width &&
          structElement[i][j] && input[testedY * pitch + testedX] > pixel)
        pixel = input[testedY * pitch + testedX];
    }
  }

  output[y * pitch + x] = pixel;
}

__global__ void removePadding(float *inputBuffer, int width, int height,
                              size_t pitch, float *outputBuffer) {
  printf("Kernel start\n");
  int line = blockDim.x * blockIdx.x + threadIdx.x;
  if (line < height)
    memcpy(outputBuffer, inputBuffer + line * pitch, width * sizeof(float));
  printf("Kernel end\n");
}

__global__ void harrisThreshold(float *harris, int width, int height,
                                float ref) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= width * height) return;

  float val = harris[i];
  harris[i] = val > ref ? val : 0;
}

struct PointCmp {
  __host__ __device__ bool operator()(float3 a, float3 b) { return a.x > b.x; }
};

__global__ void retrieveKeypoints(float *harris, float *dilatedHarris,
                                  int width, int height, size_t pitch,
                                  float3 *keypoints) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= width * height) return;

  int y = i / width;
  int x = i % width;

  float val = harris[i];
  float dilatedVal = dilatedHarris[y * pitch + x];
  float delta = abs(dilatedVal - val);

  float coef = val && (delta < __FLT_EPSILON__) ? 1 : 0;
  keypoints[i] = make_float3(harris[i] * coef, x, y);
}

float *detectHarrisPointsGPU(unsigned char **rgba_image, int width, int height,
                             size_t max_keypoints, float threshold,
                             size_t *nbFound) {
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  unsigned char *rgba_buffer;
  size_t rgba_pitch;
  rc = cudaMallocPitch(&rgba_buffer, &rgba_pitch,
                       width * RGBA_DIM * sizeof(unsigned char), height);
  if (rc) abortError("Fail buffer allocation");

  float *gray_buffer;
  size_t pitch;
  rc = cudaMallocPitch(&gray_buffer, &pitch, width * sizeof(float), height);
  if (rc) abortError("Fail buffer allocation");

  float *imx2;
  rc = cudaMallocPitch(&imx2, &pitch, width * sizeof(float), height);
  if (rc) abortError("Fail buffer allocation");

  float *imxy;
  rc = cudaMallocPitch(&imxy, &pitch, width * sizeof(float), height);
  if (rc) abortError("Fail buffer allocation");

  float *imy2;
  rc = cudaMallocPitch(&imy2, &pitch, width * sizeof(float), height);
  if (rc) abortError("Fail buffer allocation");

  float *response_flat;
  rc = cudaMalloc(&response_flat, width * height * sizeof(float));
  if (rc) abortError("Fail buffer allocation");

  float3 *keypoints;
  rc = cudaMalloc(&keypoints, width * height * sizeof(float3));
  if (rc) abortError("Fail buffer allocation");

  // Copy image to GPU memory
  rc = cudaMemcpy2D(rgba_buffer, rgba_pitch, *rgba_image, width * RGBA_DIM,
                    width * RGBA_DIM * sizeof(unsigned char), height,
                    cudaMemcpyHostToDevice);
  if (rc) abortError("Unable to copy image to GPU memory");

  // Run the kernel with blocks of size 64 x 64
  int bsize = 32;
  int w = std::ceil((float)width / bsize);
  int h = std::ceil((float)height / bsize);

  spdlog::debug("running kernel of size ({},{})", w, h);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(w, h);
  rgba2gray<<<dimGrid, dimBlock>>>(rgba_buffer, gray_buffer, width, height,
                                   rgba_pitch, pitch);

  computeDerivatives<<<dimGrid, dimBlock>>>(gray_buffer, width, height, pitch,
                                            imx2, imxy, imy2);
  if (cudaPeekAtLastError()) abortError("Computation Error");

  float *response = gray_buffer;

  computeHarrisResponse<<<dimGrid, dimBlock>>>(width, height, pitch, imx2, imxy,
                                               imy2, response);
  if (cudaPeekAtLastError()) abortError("Computation Error");

  printf("Grid size: %f\n", std::ceil(height / 1024.0));
  fflush(stdout);
  removePadding<<<(int)std::ceil(height / 1024.0), 1024>>>(response, width, height,
                                                      pitch, response_flat);
  if (cudaPeekAtLastError()) abortError("Computation Error");

  printf("Min max\n");
  fflush(stdout);

  thrust::device_vector<float> vector(response_flat,
                                      response_flat + (width * height));

  printf("Min max 2\n");
  fflush(stdout);

  thrust::pair minmax = thrust::minmax_element(vector.begin(), vector.end());

  printf("Min max 3\n");
  fflush(stdout);

  float min = *(minmax.first);
  float max = *(minmax.second);

  float ref = min + threshold * (max - min);

  harrisThreshold<<<std::ceil(width * height / 1024.0), 1024>>>(
      response_flat, width, height, ref);
  if (cudaPeekAtLastError()) abortError("Computation Error");

  float *dilated = imx2;

  morphoDilate<<<dimGrid, dimBlock>>>(response, width, height, pitch, dilated);

  if (cudaPeekAtLastError()) abortError("Computation Error");

  retrieveKeypoints<<<std::ceil(width * height / 1024.0), 1024>>>(
      response_flat, dilated, width, height, pitch, keypoints);
  if (cudaPeekAtLastError()) abortError("Computation Error");

  thrust::device_vector<float3> keypoints_vec(keypoints,
                                              keypoints + (width * height));

  thrust::sort(keypoints_vec.begin(), keypoints_vec.end(), PointCmp());

  size_t limit = 0;
  while (limit < keypoints_vec.size() && limit < max_keypoints && ((float3)keypoints_vec[limit]).x != 0) limit++;

  if (limit < keypoints_vec.size() && limit > 0 && ((float3)keypoints_vec[limit]).x == 0) limit--;


  printf("Results allocation\n");
  fflush(stdout);

  float *result;
  rc = cudaMallocHost(&result, width * height * sizeof(float3));
  if (rc) abortError("Fail buffer allocation");

  printf("Results copy\n");
  fflush(stdout);

  rc = cudaMemcpy(result, keypoints_vec.data().get(), limit * sizeof(float3), cudaMemcpyDeviceToHost);
  if (rc) abortError("Fail device to host copy");
  *nbFound = limit + 1;

  printf("Free\n");
  fflush(stdout);

  // Free
  rc = cudaFree(rgba_buffer);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(gray_buffer);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(imx2);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(imxy);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(imy2);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(response_flat);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(keypoints);
  if (rc) abortError("Unable to free memory");

  return result;
}
