#pragma once

#include "Pixel.h"

#ifndef EXP_CUDA_GRAYSCALE

	#ifndef BUILD_CUDA_GRAYSCALE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Cuda.lib")
		#define EXP_CUDA_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_CUDA_GRAYSCALE __declspec(dllexport)
	#endif // !BUILD_CUDA_GRAYSCALE

#endif // !EXP_CUDA_GRAYSCALE

extern "C" EXP_CUDA_GRAYSCALE bool grayScaleWithCuda(Pixel_t * image, int pixels_to_process);
