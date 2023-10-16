#pragma once

#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaKernel.h"
#include "Pixel.h"

__global__ void complementKernel(Pixel_t* image) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Pixel_t inpixel = *(image + i);
	inpixel.R = 255 - inpixel.R;
	inpixel.G = 255 - inpixel.G;
	inpixel.B = 255 - inpixel.B;
	*(image + i) = inpixel;
}

bool complementCuda(Pixel_t* image, int pixels_to_process)
{	
	cudaError_t cudaStatus;
	Pixel_t* dev_image;	
	
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

	complementKernel<<<gridSize,blockSize>>>(dev_image);

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
	cudaFree(dev_image);
	return false;
}
