#include <tuple>
#include <vector>

float** compute_harris_response(unsigned char **img, int width, int height);
std::vector<std::tuple<float, int, int>> detectHarrisPoints(unsigned char **image, int width, int height, int max_keypoints, float threshold);