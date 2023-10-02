#pragma once

#include <stdio.h>
#include <string>

#ifndef EXP_CUDA_GRAYSCALE

	#ifndef BUILD_CUDA_GRAYSCALE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Cuda.lib")
		#define EXP_CUDA_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_CUDA_GRAYSCALE __declspec(dllexport)
	#endif // !BUILD_CUDA_GRAYSCALE

#endif // !EXP_CUDA_GRAYSCALE

extern "C" EXP_CUDA_GRAYSCALE bool grayScaleWithCuda(std::string file_src, std::string file_dst);
