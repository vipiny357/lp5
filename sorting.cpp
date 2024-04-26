#include <omp.h>

#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace chrono;

// Function to perform bubble sort
void bubbleSort(vector<int>& arr) {
	int n = arr.size();
	for (int i = 0; i < n - 1; i++) {
		for (int j = 0; j < n - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				swap(arr[j], arr[j + 1]);
			}
		}
	}
}

// Function to perform parallel bubble sort using Odd-even transposition sort
void parallelBubbleSort(vector<int>& arr) {
	int n = arr.size();
	bool sorted = false;
	while (!sorted) {
		sorted = true;
		#pragma omp parallel for shared(arr, sorted)
		for (int i = 0; i < n - 1; i++) {
			if (arr[i] > arr[i + 1]) {
				swap(arr[i], arr[i + 1]);
				sorted = false;
			}
			i++;
			if (arr[i] > arr[i + 1]) {
				swap(arr[i], arr[i + 1]);
				sorted = false;
			}
		}
		#pragma omp barrier
	}
}

// Function to perform merge sort
void merge(vector<int>& arr, int l, int m, int r) {
	int n1 = m - l + 1;
	int n2 = r - m;
	vector<int> L(n1), R(n2);

	for (int i = 0; i < n1; i++) L[i] = arr[l + i];
	for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

	int i = 0, j = 0, k = l;
	while (i < n1 && j < n2) {
		if (L[i] <= R[j]) {
			arr[k] = L[i];
			i++;
		} else {
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	while (i < n1) {
		arr[k] = L[i];
		i++;
		k++;
	}

	while (j < n2) {
		arr[k] = R[j];
		j++;
		k++;
	}
}

void mergeSort(vector<int>& arr, int l, int r) {
	if (l < r) {
		int m = l + (r - l) / 2;
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}

void parallelMergeSort(vector<int>& arr, int n) {
	#pragma omp parallel
	{
		#pragma omp single
		{
			mergeSort(arr, 0, n - 1);
		}
	}
}

void init_arr(vector<int>& arr, int n) {
	// Initialize array with random values
	srand(time(0));
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % n;
	}
}

int main() {
	int n = 10000;	// Number of elements in the array
	vector<int> arr(n);

	// Initialize array with random values
	init_arr(arr, n);

	// Measure time for sequential bubble sort
	auto start = high_resolution_clock::now();
	bubbleSort(arr);
	auto end = high_resolution_clock::now();
	auto durationSeqBubbleSort = duration_cast<milliseconds>(end - start);

	cout << "Time taken for sequential bubble sort: "
		 << durationSeqBubbleSort.count() << " milliseconds" << endl;

	// Initialize array with random values
	init_arr(arr, n / 2);

	// Measure time for parallel bubble sort
	start = high_resolution_clock::now();
	parallelBubbleSort(arr);
	end = high_resolution_clock::now();
	auto durationParBubbleSort = duration_cast<milliseconds>(end - start);

	cout << "Time taken for parallel bubble sort: "
		 << durationParBubbleSort.count() << " milliseconds" << endl;

	// Initialize array with random values
	init_arr(arr, n);

	// Measure time for sequential merge sort
	start = high_resolution_clock::now();
	mergeSort(arr, 0, n - 1);
	end = high_resolution_clock::now();
	auto durationSeqMergeSort = duration_cast<milliseconds>(end - start);

	cout << "Time taken for sequential merge sort: "
		 << durationSeqMergeSort.count() << " milliseconds" << endl;

	// Initialize array with random values
	// init_arr(arr, n);

	// Measure time for parallel merge sort
	start = high_resolution_clock::now();
	parallelMergeSort(arr, n);
	end = high_resolution_clock::now();
	auto durationParMergeSort = duration_cast<milliseconds>(end - start);

	cout << "Time taken for parallel merge sort: "
		 << durationParMergeSort.count() << " milliseconds" << endl;

	return 0;
}
