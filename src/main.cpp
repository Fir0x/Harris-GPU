#include <png.h>

#include <CLI/CLI.hpp>
#include <iostream>

#include "CPU/harris.hpp"
#include "GPU/harris.hpp"

#define MAX_KEYPOINTS 2000
#define THRESHOLD 0.5

png_bytepp read_png(const std::string file_name, int *width, int *height) {
  FILE *fp = fopen(file_name.c_str(), "rb");
  if (fp == NULL) return nullptr;

  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  auto info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);

  png_read_info(png_ptr, info_ptr);

  *width = png_get_image_width(png_ptr, info_ptr);
  *height = png_get_image_height(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  // Format dark magic
  if (bit_depth == 16) png_set_strip_16(png_ptr);

  if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);

  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png_ptr);

  if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);

  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png_ptr);

  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(sizeof(png_bytep) * (*height));
  for (int y = 0; y < *height; y++) {
    row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(fp);

  return row_pointers;
}

png_bytepp rgb2gray(png_bytepp img, int w, int h) {
  png_bytepp result = (png_bytepp)malloc(h * sizeof(unsigned char *));
  for (int y = 0; y < h; y++) {
    result[y] = (unsigned char *)malloc(w * sizeof(unsigned char));
    for (int x = 0; x < w; x++) {
      // OpenCv grayscale: 0.299 R + 0.587 G + 0.114 B
      result[y][x] = 0.299 * img[y][x * 4] + 0.587 * img[y][x * 4 + 1] +
                     0.114 * img[y][x * 4 + 2];
    }
  }

  return result;
}

png_bytepp gray2rgb(png_bytepp img, int w, int h) {
  png_bytepp result = (png_bytepp)malloc(h * sizeof(unsigned char *));
  for (int y = 0; y < h; y++) {
    result[y] = (unsigned char *)malloc(w * 4 * sizeof(unsigned char));
    for (int x = 0; x < w; x++) {
      result[y][x * 4] = img[y][x];
      result[y][x * 4 + 1] = img[y][x];
      result[y][x * 4 + 2] = img[y][x];
      result[y][x * 4 + 3] = 0xFF;
    }
  }

  return result;
}

unsigned char *flatten(unsigned char **image, int width, int height,
                       int depth) {
  unsigned char *image_flat =
      (unsigned char *)malloc(height * width * depth * sizeof(unsigned char));

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width * depth; x++)
      image_flat[y * width * depth + x] = image[y][x];

  return image_flat;
}

png_bytepp reshape(unsigned char *array, int width, int height, int depth) {
  png_bytepp array_2D = (png_bytepp)malloc(height * sizeof(unsigned char *));
  for (int y = 0; y < height; y++)
    array_2D[y] =
        (unsigned char *)malloc(width * depth * sizeof(unsigned char));

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width * depth; x++)
      array_2D[y][x] = array[y * width * depth + x];

  return array_2D;
}

void free_image(png_bytepp img, int h) {
  for (int i = 0; i < h; i++) free(img[i]);

  free(img);
}

void write_png(const png_bytepp buffer, int width, int height,
               const char *filename) {
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr) return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE *fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);

  png_write_image(png_ptr, buffer);

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
}

void drawHarrisPoints(png_bytepp image, int width, int height,
                      std::vector<std::tuple<float, int, int>> keypoints,
                      int pointSize) {
  for (const auto &kp : keypoints) {
    int y = std::get<1>(kp);
    int x = std::get<2>(kp);

    for (int i = (y - pointSize / 2); i < height && i <= (y + pointSize / 2);
         i++) {
      if (i < 0) continue;

      for (int j = (x - pointSize / 2); j < width && j <= (x + pointSize / 2);
           j++) {
        if (j >= 0) {
          image[i][j * 4] = 0;
          image[i][j * 4 + 1] = 255;
          image[i][j * 4 + 2] = 0;
          image[i][j * 4 + 3] = 255;
        }
      }
    }
  }
}

void drawGPUHarrisPoints(png_bytepp image, int width, int height,
                         float *keypoints, size_t nbFound, int pointSize) {
  for (size_t i = 0; i < nbFound; i++) {
    int x = keypoints[i * 3 + 1];
    int y = keypoints[i * 3 + 2];

    for (int i = (y - pointSize / 2); i < height && i <= (y + pointSize / 2);
         i++) {
      if (i < 0) continue;

      for (int j = (x - pointSize / 2); j < width && j <= (x + pointSize / 2);
           j++) {
        if (j >= 0) {
          image[i][j * 4] = 0;
          image[i][j * 4 + 1] = 255;
          image[i][j * 4 + 2] = 0;
          image[i][j * 4 + 3] = 255;
        }
      }
    }
  }
}

void matrix2image(float **m, int width, int height, const char *filename) {
  float minVal = m[0][0];
  float maxVal = m[0][0];
  png_bytepp img = (png_bytepp)malloc(height * sizeof(png_bytepp));
  for (int y = 0; y < height; y++) {
    img[y] = (png_bytep)malloc(width * 4 * sizeof(png_bytep));
    for (int x = 0; x < width; x++) {
      float val = m[y][x];
      if (val < minVal)
        minVal = m[y][x];
      else if (val > maxVal)
        maxVal = val;
    }
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char val =
          (unsigned char)(255.0 * (m[y][x] - minVal) / (maxVal - minVal));
      img[y][x * 4] = val;
      img[y][x * 4 + 1] = val;
      img[y][x * 4 + 2] = val;
      img[y][x * 4 + 3] = 255;
    }
  }

  write_png(img, width, height, filename);
}

void flatMatrix2Image(float *m, int width, int height, const char *filename) {
  float minVal = m[0];
  float maxVal = m[0];
  png_bytepp img = (png_bytepp)malloc(height * sizeof(png_bytepp));
  for (int y = 0; y < height; y++) {
    img[y] = (png_bytep)malloc(width * 4 * sizeof(png_bytep));
    for (int x = 0; x < width; x++) {
      float val = m[y * width + x];
      if (val < minVal)
        minVal = m[y * width + x];
      else if (val > maxVal)
        maxVal = val;
    }
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char val = (unsigned char)(255.0 * (m[y * width + x] - minVal) /
                                          (maxVal - minVal));
      img[y][x * 4] = val;
      img[y][x * 4 + 1] = val;
      img[y][x * 4 + 2] = val;
      img[y][x * 4 + 3] = 255;
    }
  }

  write_png(img, width, height, filename);
}

void debugSteps(png_bytepp gray, int width, int height) {
  float **dX;
  float **dY;
  gauss_derivatives(gray, width, height, &dX, &dY);
  matrix2image(dX, width, height, "debug/dX.png");
  matrix2image(dY, width, height, "debug/dY.png");

  float **resp = computeHarrisResponse(gray, width, height);
  matrix2image(resp, width, height, "debug/harrisResp.png");

  png_bytepp erod = morphoErode(gray, width, height);
  write_png(gray2rgb(erod, width, height), width, height, "debug/eroded.png");

  float **dilat = morphoDilate(resp, width, height);
  matrix2image(dilat, width, height, "debug/dilated.png");

  png_bytepp thres = harrisThreshold(resp, width, height, 0.5);
  write_png(gray2rgb(thres, width, height), width, height, "debug/thres.png");
}

int main(int argc, char **argv) {
  std::string inputFile;
  std::string filename = "output.png";
  std::string mode = "CPU";

  CLI::App app{"harris"};
  app.add_option("-o", filename, "Output image");
  app.add_option("image", inputFile, "Input image");
  app.add_set("-m", mode, {"GPU", "CPU", "BENCH"},
              "Either 'GPU', 'CPU' or 'BENCH'");

  CLI11_PARSE(app, argc, argv);

  int imageWidth;
  int imageHeight;
  png_bytepp image = read_png(inputFile, &imageWidth, &imageHeight);

  if (image == nullptr) {
    std::cout << "Could not read the image: " << inputFile << std::endl;
    return 1;
  }

  std::vector<std::tuple<float, int, int>> keypoints;

  if (mode == "CPU") {
    png_bytepp gray = rgb2gray(image, imageWidth, imageHeight);
    keypoints = detectHarrisPoints(gray, imageWidth, imageHeight, MAX_KEYPOINTS,
                                   THRESHOLD);

    free_image(gray, imageHeight);

    std::cout << keypoints.size() << " keypoints retrieved\n";

    drawHarrisPoints(image, imageWidth, imageHeight, keypoints, 5);
  } else if (mode == "GPU") {
    unsigned char *image_1D = flatten(image, imageWidth, imageHeight, 4);
    size_t nbFound;
    float *keypoints = detectHarrisPointsGPU(
        &image_1D, imageWidth, imageHeight, MAX_KEYPOINTS, THRESHOLD, &nbFound);

    drawGPUHarrisPoints(image, imageWidth, imageHeight, keypoints, nbFound, 5);

    std::cout << nbFound << " keypoints retrieved\n";
  } else if (mode == "BENCH") {
    std::cout << "Image size : " << imageWidth << "x" << imageHeight << "\n";

    // using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    start = std::chrono::system_clock::now();

    png_bytepp gray = rgb2gray(image, imageWidth, imageHeight);
    keypoints = detectHarrisPoints(gray, imageWidth, imageHeight,
                                    MAX_KEYPOINTS, THRESHOLD);
    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;

    std::cout << "CPU: " << elapsed_seconds.count() << "s - "
              << keypoints.size() << " keypoints retrieved\n";

    start = std::chrono::system_clock::now();

    size_t nbFound;
    unsigned char *image_1D = flatten(image, imageWidth, imageHeight, 4);
    float *keypoints = detectHarrisPointsGPU(&image_1D, imageWidth, imageHeight, MAX_KEYPOINTS,
                          THRESHOLD, &nbFound);

    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;

    std::cout << "GPU: " << elapsed_seconds.count() << "s - " << nbFound
              << " keypoints retrieved\n";

    drawGPUHarrisPoints(image, imageWidth, imageHeight, keypoints, nbFound, 5);
  }

  write_png(image, imageWidth, imageHeight, filename.c_str());
  std::cout << "Result in " << filename << "\n";

  free_image(image, imageHeight);

  return 0;
}