#include <spdlog/spdlog.h>

#include <cassert>
#include <png.h>

#include "harris.hpp"

#define RGBA_DIM 4

[[gnu::noinline]] void _abortError(const char* msg, const char* fname,
                                   int line) {
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

// Device code
__global__ void rgba2gray(unsigned char *rgba_buffer, unsigned char *gray_buffer, int width, int height, size_t rgba_pitch, size_t gray_pitch) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int y_rgba = y * rgba_pitch;
  gray_buffer[y * gray_pitch + x] = 0.299 * rgba_buffer[y_rgba + x*4] + 0.587 * rgba_buffer[y_rgba + x*4 + 1] + 0.114 * rgba_buffer[y_rgba + x*4 + 2];
}

void detectHarrisPointsGPU(unsigned char **rgba_image, unsigned char **gray_image, int width, int height, size_t max_keypoints, float threshold) {
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  unsigned char *rgba_buffer;
  size_t rgba_pitch;
  rc = cudaMallocPitch(&rgba_buffer, &rgba_pitch, width * RGBA_DIM * sizeof(unsigned char), height);
  if (rc) abortError("Fail buffer allocation");

  unsigned char *gray_buffer;
  size_t gray_pitch;
  rc = cudaMallocPitch(&gray_buffer, &gray_pitch, width * sizeof(unsigned char), height);
  if (rc) abortError("Fail buffer allocation");

  // Copy image to GPU memory
  rc = cudaMemcpy2D(rgba_buffer, rgba_pitch, *rgba_image, width * RGBA_DIM,
                    width * RGBA_DIM * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
  if (rc) abortError("Unable to copy image to GPU memory");

  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    rgba2gray<<<dimGrid, dimBlock>>>(rgba_buffer, gray_buffer, width, height, rgba_pitch, gray_pitch);

    if (cudaPeekAtLastError()) abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(*gray_image, width, gray_buffer, gray_pitch,
                    width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
  if (rc) abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(rgba_buffer);
  if (rc) abortError("Unable to free memory");

  rc = cudaFree(gray_buffer);
  if (rc) abortError("Unable to free memory");
}