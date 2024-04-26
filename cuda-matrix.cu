#include <iostream>
#include <chrono>

// CUDA Runtime
#include <cuda_runtime.h>

#define N 1024 // Matrix size

// CUDA kernel to perform matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int n) {
	// Calculate thread index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < n && col < n) {
		int sum = 0;
		for (int i = 0; i < n; ++i) {
			sum += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = sum;
	}
}

// Function to perform matrix multiplication sequentially
void matrixMulSeq(int *a, int *b, int *c, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int sum = 0;
			for (int k = 0; k < n; ++k) {
				sum += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = sum;
		}
	}
}

int main() {
	int *a, *b, *c; // Host matrices
	int *d_a, *d_b, *d_c; // Device matrices

	// Memory allocation on the host
	a = new int[N * N];
	b = new int[N * N];
	c = new int[N * N];

	// Initialize matrices a and b
	for (int i = 0; i < N * N; ++i) {
		a[i] = i;
		b[i] = i;
	}

	// Memory allocation on the device
	cudaMalloc(&d_a, N * N * sizeof(int));
	cudaMalloc(&d_b, N * N * sizeof(int));
	cudaMalloc(&d_c, N * N * sizeof(int));

	// Copy matrices a and b from host to device
	cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

	// Define block and grid dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Start time for parallel version
	auto start_parallel = std::chrono::high_resolution_clock::now();

	// Call the kernel
	matrixMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

	// End time for parallel version
	auto end_parallel = std::chrono::high_resolution_clock::now();

	// Copy the result matrix from device to host
	cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	// Calculate time taken for parallel version
	std::chrono::duration<double> elapsed_parallel = end_parallel - start_parallel;
	std::cout << "Parallel version took " << elapsed_parallel.count() << " seconds." << std::endl;

	// Start time for sequential version
	auto start_sequential = std::chrono::high_resolution_clock::now();

	// Perform matrix multiplication sequentially
	matrixMulSeq(a, b, c, N);

	// End time for sequential version
	auto end_sequential = std::chrono::high_resolution_clock::now();

	// Calculate time taken for sequential version
	std::chrono::duration<double> elapsed_sequential = end_sequential - start_sequential;
	std::cout << "Sequential version took " << elapsed_sequential.count() << " seconds." << std::endl;

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Free host memory
	delete[] a;
	delete[] b;
	delete[] c;
}
