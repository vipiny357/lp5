#include <iostream>
#include <vector>
#include <chrono>

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	// Size of the vectors
	int n = 1'000'000; // Adjust the size as needed

	// Initialize host vectors
	std::vector<int> h_a(n);
	std::vector<int> h_b(n);
	std::vector<int> h_c(n);

	// Initialize device vectors
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, n * sizeof(int));
	cudaMalloc(&d_b, n * sizeof(int));
	cudaMalloc(&d_c, n * sizeof(int));

	// Populate host vectors with random values
	for (int i = 0; i < n; ++i) {
		h_a[i] = rand() % 100;
		h_b[i] = rand() % 100;
	}

	// Copy host vectors to device
	cudaMemcpy(d_a, h_a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	// Define block size and grid size
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	// Sequential vector addition for comparison
	auto start_sequential = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n; ++i) {
		h_c[i] = h_a[i] + h_b[i];
	}
	auto end_sequential = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> sequential_time = end_sequential - start_sequential;

	// Parallel vector addition using CUDA
	auto start_parallel = std::chrono::high_resolution_clock::now();
	vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
	cudaDeviceSynchronize();
	auto end_parallel = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> parallel_time = end_parallel - start_parallel;

	// Copy result from device to host
	cudaMemcpy(h_c.data(), d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Check for errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
	}

	// Output timings
	std::cout << "Sequential Time: " << sequential_time.count() << " seconds" << std::endl;
	std::cout << "Parallel Time: " << parallel_time.count() << " seconds" << std::endl;

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}