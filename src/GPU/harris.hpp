#pragma once
#include <cstddef>
#include <memory>

float *detectHarrisPointsGPU(unsigned char **rgba_image, int width, int height,
                              size_t max_keypoints, float threshold,
                              size_t *nbFound);
