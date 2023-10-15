#pragma once

#define BUILD_CUDA_GRAYSCALE

#include <stdio.h>
#include <string>

#include "Halide.h"
#include "halide_image_io.h"

#include "GrayScaleCuda.h"
#include "CudaKernel.h"
#include "Pixel.h"

#include <filesystem>

using namespace Halide;
using namespace Halide::Tools;

extern "C" EXP_CUDA_GRAYSCALE bool grayScaleWithCuda(std::string file_src, std::string file_dst)
{
	Halide::Buffer<uint8_t> input = load_image(file_src);
	std::filesystem::remove(file_dst);
	Halide::Buffer<uint8_t> output(input.width(), input.height(), input.channels());
	int pixels_to_process = input.height() * input.width();

	Pixel_t* image = (Pixel_t*)malloc(pixels_to_process * sizeof(Pixel_t));	
	for (int x = 0, p = 0; x < input.width(); x++)
	{
		for (int y = 0; y < input.height(); y++)
		{
			Pixel_t inpixel = *(image + p);
			inpixel.R = input(x, y, 0);
			inpixel.G = input(x, y, 1);
			inpixel.B = input(x, y, 2);
			*(image + p) = inpixel;
			p++;
		}
	}

	if(!grayScaleCuda(image, pixels_to_process))
		goto Error;	

	for (int x = 0, p = 0; x < output.width(); x++)
	{
		for (int y = 0; y < output.height(); y++)
		{
			Pixel_t inpixel = *(image + p);
			output(x, y, 0) = inpixel.R;
			output(x, y, 1) = inpixel.G;
			output(x, y, 2) = inpixel.B;
			*(image + p) = inpixel;
			p++;
		}
	}

	save_image(output, file_dst);	
	free(image);
	return true;

Error:
	free(image);
	return false;
}