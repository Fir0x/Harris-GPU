#include <CLI/CLI.hpp>
#include <png.h>
#include <iostream>

#include "CPU/harris.hpp"

png_bytepp read_png(const std::string file_name, int *width, int *height)
{
    FILE *fp = fopen(file_name.c_str(), "rb");
    if (fp == NULL)
        return nullptr;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    auto info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);

    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    // Format dark magic
    if(bit_depth == 16)
        png_set_strip_16(png_ptr);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if(png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    if(color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    png_bytepp row_pointers = (png_bytepp)malloc(sizeof(png_bytep) * (*height));
    for(int y = 0; y < *height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
    fclose(fp);

    return row_pointers;
}

png_bytepp rgb2gray(png_bytepp img, int w, int h)
{
    png_bytepp result = (png_bytepp)malloc(h * sizeof(unsigned char*));
    for (int y = 0; y < h; y++)
    {
        result[y] = (unsigned char*)malloc(w * sizeof(unsigned char));
        for (int x  = 0; x < w; x++)
            result[y][x] = (img[y][x*4] + img[y][x*4+1] + img[y][x*4+2]) / 3;
    }

    return result;
}

png_bytepp gray2rgb(png_bytepp img, int w, int h)
{
    png_bytepp result = (png_bytepp)malloc(h * sizeof(unsigned char*));
    for (int y = 0; y < h; y++)
    {
        result[y] = (unsigned char*)malloc(w * 4 * sizeof(unsigned char));
        for (int x = 0; x < w; x++)
        {
            result[y][x*4] = img[y][x];
            result[y][x*4+1] = img[y][x];
            result[y][x*4+2] = img[y][x];
            result[y][x*4+3] = 0xFF;
        }
    }

    return result;
}

void free_image(png_bytepp img, int h)
{
    for (int i = 0; i < h; i++)
        free(img[i]);

    free(img);
}

void write_png(const png_bytepp buffer,
               int width,
               int height,
               const char* filename)
{
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png_ptr)
        return;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, nullptr);
        return;
    }

    FILE* fp = fopen(filename, "wb");
    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr,
                width,
                height,
                8,
                PNG_COLOR_TYPE_RGB_ALPHA,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    png_write_image(png_ptr, buffer);

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void drawHarrisPoints(png_bytepp image, int width, int height, std::vector<std::tuple<float, int, int>> keypoints, int pointSize)
{
    for (const auto &kp : keypoints)
    {
        int y = std::get<1>(kp);
        int x = std::get<2>(kp);

        for (int i = (y - pointSize/2); i < height && i <= (y + pointSize/2); i++)
        {
            if (i < 0)
                continue;

            for (int j = (x - pointSize/2); j < width && j <= (x + pointSize/2); j++)
            {
                if (j >= 0)
                {
                    image[i][j*4] = 0;
                    image[i][j*4+1] = 255;
                    image[i][j*4+2] = 0;
                    image[i][j*4+3] = 255;
                }
            }
        }
    }
}

void matrix2image(float **m, int width, int height, char *filename)
{
    int minVal = m[0][0];
    int maxVal = m[0][0];
    png_bytepp img = (png_bytepp)malloc(height * sizeof(png_bytepp));
    for (int y = 0; y < height; y++)
    {
        img[y] = (png_bytep)malloc(width * 4 * sizeof(png_bytep));
        for (int x = 0; x < width; x++)
        {
            float val = m[y][x];
            if (val < minVal)
                minVal = m[y][x];
            else if (val > maxVal)
                maxVal = val;
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char val = (unsigned char)(255.0 * (m[y][x] - minVal) / (maxVal - minVal));
            img[y][x*4] = val;
            img[y][x*4+1] = val;
            img[y][x*4+2] = val;
            img[y][x*4+3] = 255;
        }
    }

    write_png(img, width, height, filename);
}

void debugSteps(png_bytepp gray, int width, int height)
{
    float **dX;
    float **dY;
    gauss_derivatives(gray, width, height, &dX, &dY);

    float **resp = computeHarrisResponse(gray, width, height);
    png_bytepp erod = morphoErode(gray, width, height);
    float **dilat = morphoDilate(resp, width, height);
    png_bytepp thres = harrisThreshold(resp, width, height, 0.5);

    matrix2image(dX, width, height, "debug/dX.png");
    matrix2image(dY, width, height, "debug/dY.png");
    matrix2image(resp, width, height, "debug/harrisResp.png");
    write_png(gray2rgb(erod, width, height), width, height, "debug/eroded.png");
    matrix2image(dilat, width, height, "debug/dilated.png");
    write_png(gray2rgb(thres, width, height), width, height, "debug/thres.png");
}

int main(int argc, char** argv)
{
    std::string inputFile;
    std::string filename = "output.png";
    std::string mode = "GPU";

    CLI::App app{"harris"};
    app.add_option("-o", filename, "Output image");
    app.add_option("image", inputFile, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

    CLI11_PARSE(app, argc, argv);

    int imageWidth;
    int imageHeight;
    png_bytepp image = read_png(inputFile, &imageWidth, &imageHeight);
    if(image == nullptr)
    {
        std::cout << "Could not read the image: " << inputFile << std::endl;
        return 1;
    }

    png_bytepp gray = rgb2gray(image, imageWidth, imageHeight);

    debugSteps(gray, imageWidth, imageHeight);

    auto keypoints = detectHarrisPoints(gray, imageWidth, imageHeight, 2000, 0.5);
    std::cout << keypoints.size() << " keypoints retrieved\n";
    for (int i = 0; i < 10 && keypoints.size(); i++)
    {
        std::cout << "X:" << std::get<2>(keypoints[i]) << "Y:" << std::get<1>(keypoints[i]) << "\n";
    }
    png_bytepp rgbGray = gray2rgb(gray, imageWidth, imageHeight);
    drawHarrisPoints(rgbGray, imageWidth, imageHeight, keypoints, 5);
    write_png(rgbGray, imageWidth, imageHeight, "converted.png");

    free_image(image, imageHeight);
    free_image(gray, imageHeight);
    free_image(rgbGray, imageHeight);

    return 0;
}