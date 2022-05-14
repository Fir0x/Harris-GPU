#include <tuple>
#include <vector>

void gauss_derivatives(unsigned char **img, int  width, int height, float ***dX, float ***dY);
float **computeHarrisResponse(unsigned char **img, int width, int height);
float** compute_harris_response(unsigned char **img, int width, int height);
std::vector<std::tuple<float, int, int>> detectHarrisPoints(unsigned char **image, int width, int height, int max_keypoints, float threshold);