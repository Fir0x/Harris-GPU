#pragma once
#include <cstddef>
#include <memory>
 
void detectHarrisPointsGPU(unsigned char **rgba_image, unsigned char **gray_image, int width, int height, size_t max_keypoints, float threshold);
