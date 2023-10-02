#define BUILD_CUDA_GRAYSCALE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <string>

#include "GrayScaleCuda.h"
#include "Pixel.h"

__global__ void grayScaleKernel(Pixel_t* image) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    Pixel_t inpixel = *(image + i);
    int value = (0.299 * inpixel.R) + (0.587 * inpixel.G) + (0.114 * inpixel.B);
    inpixel.R = value;
    inpixel.G = value;
    inpixel.B = value;
    *(image + i) = inpixel;
}

extern "C" EXP_CUDA_GRAYSCALE bool grayScaleWithCuda(std::string file_src, std::string file_dst)
{
	Pixel_t* dev_image = 0;
	cudaError_t cudaStatus;

	/*
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_image, pixels_to_process * sizeof(Pixel_t));
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_image, image, pixels_to_process * sizeof(Pixel_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	int blockSize, gridSize;
	blockSize = 256;
	gridSize = (int)ceil((float)pixels_to_process / blockSize);

	grayScaleKernel<<<gridSize,blockSize>>>(dev_image);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(image, dev_image, pixels_to_process * sizeof(Pixel_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	return true;

Error:
	cudaFree(dev_image); */
	return false;
}
