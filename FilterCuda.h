#pragma once

#include <stdio.h>
#include <string>

#include "Pixel.h"

#ifndef EXP_CUDA_GRAYSCALE

	#ifndef BUILD_CUDA
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Cuda.lib")
		#define EXP_CUDA_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_CUDA_GRAYSCALE __declspec(dllexport)
	#endif

#endif

extern "C" EXP_CUDA_GRAYSCALE bool grayScaleWithCuda(std::string file_src, std::string file_dst);

#ifndef EXP_CUDA_COMPLEMENT

	#ifndef BUILD_CUDA
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Cuda.lib")
		#define EXP_CUDA_COMPLEMENT __declspec(dllimport)
	#else
		#define EXP_CUDA_COMPLEMENT __declspec(dllexport)
	#endif

#endif

extern "C" EXP_CUDA_COMPLEMENT bool complementWithCuda(std::string file_src, std::string file_dst);
