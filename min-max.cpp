#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h> // OpenMP library for parallelization

using namespace std;
using namespace chrono;

// Function to find minimum element
int parallelMin(const vector<int>& data) {
	int min_val = data[0];
	#pragma omp parallel for reduction(min:min_val)
	for (size_t i = 0; i < data.size(); ++i) {
		if (data[i] < min_val) {
			min_val = data[i];
		}
	}
	return min_val;
}

// Function to find maximum element
int parallelMax(const vector<int>& data) {
	int max_val = data[0];
	#pragma omp parallel for reduction(max:max_val)
	for (size_t i = 0; i < data.size(); ++i) {
		if (data[i] > max_val) {
			max_val = data[i];
		}
	}
	return max_val;
}

// Function to calculate sum of elements
int parallelSum(const vector<int>& data) {
	int sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (size_t i = 0; i < data.size(); ++i) {
		sum += data[i];
	}
	return sum;
}

// Function to calculate average of elements
double parallelAverage(const vector<int>& data) {
	int sum = parallelSum(data);
	return static_cast<double>(sum) / data.size();
}

// Sequential versions for comparison
int sequentialMin(const vector<int>& data) {
	return *min_element(data.begin(), data.end());
}

int sequentialMax(const vector<int>& data) {
	return *max_element(data.begin(), data.end());
}

int sequentialSum(const vector<int>& data) {
	return accumulate(data.begin(), data.end(), 0);
}

double sequentialAverage(const vector<int>& data) {
	int sum = sequentialSum(data);
	return static_cast<double>(sum) / data.size();
}

void init_arr(vector<int>& arr, int n) {
	// Initialize array with random values
	srand(time(0));
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % n;
	}
}

int main() {
	int n = 10'000'000;
	vector<int> data(n, 1); // Example data, 10 million elements with value 1
	init_arr(data, n);

	auto start = high_resolution_clock::now();
	int min_val_parallel = parallelMin(data);
	auto end = high_resolution_clock::now();
	auto duration_parallel = duration_cast<milliseconds>(end - start);
	cout << "  Parallel Min: " << duration_parallel.count() << " milliseconds" << endl;
	
	start = high_resolution_clock::now();
	int min_val_sequential = sequentialMin(data);
	end = high_resolution_clock::now();
	auto duration_sequential = duration_cast<milliseconds>(end - start);
	cout << "Sequential Min: " << duration_sequential.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	int max_val_parallel = parallelMax(data);
	end = high_resolution_clock::now();
	auto duration_parallel_max = duration_cast<milliseconds>(end - start);
	cout << "  Parallel Max: " << duration_parallel_max.count() << " milliseconds" << endl;
	
	start = high_resolution_clock::now();
	int max_val_sequential = sequentialMax(data);
	end = high_resolution_clock::now();
	auto duration_sequential_max = duration_cast<milliseconds>(end - start);
	cout << "Sequential Max: " << duration_sequential_max.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	int sum_parallel = parallelSum(data);
	end = high_resolution_clock::now();
	auto duration_parallel_sum = duration_cast<milliseconds>(end - start);
	cout << "  Parallel Sum: " << duration_parallel_sum.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	int sum_sequential = sequentialSum(data);
	end = high_resolution_clock::now();
	auto duration_sequential_sum = duration_cast<milliseconds>(end - start);
	cout << "Sequential Sum: " << duration_sequential_sum.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	double avg_parallel = parallelAverage(data);
	end = high_resolution_clock::now();
	auto duration_parallel_avg = duration_cast<milliseconds>(end - start);
	cout << "  Parallel Avg: " << duration_parallel_avg.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	double avg_sequential = sequentialAverage(data);
	end = high_resolution_clock::now();
	auto duration_sequential_avg = duration_cast<milliseconds>(end - start);
	cout << "Sequential Avg: " << duration_sequential_avg.count() << " milliseconds" << endl;

}