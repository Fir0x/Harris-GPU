#include <tuple>
#include <vector>

void gauss_derivatives(unsigned char **img, int  width, int height, float ***dX, float ***dY);
unsigned char** morphoErode(unsigned char** img, int width, int height);
float **morphoDilate(float **img, int width, int height);
float **computeHarrisResponse(unsigned char **img, int width, int height);
unsigned char** harrisThreshold(float **harris, int width, int height, float threshold);
std::vector<std::tuple<float, int, int>> detectHarrisPoints(unsigned char **image, int width, int height, int max_keypoints, float threshold);