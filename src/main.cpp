#include <CLI/CLI.hpp>
#include <png.h>
#include <iostream>

#include "CPU/harris.hpp"

png_bytepp read_png(const std::string file_name)
{
    FILE *fp = fopen(file_name.c_str(), "rb");
    if (fp == NULL)
        return nullptr;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    auto info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    auto row_pointers = png_get_rows(png_ptr, info_ptr);
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);

    return row_pointers;
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

    png_bytepp image = read_png(inputFile);
    if(image == nullptr)
    {
        std::cout << "Could not read the image: " << inputFile << std::endl;
        return 1;
    }

    return 0;
}