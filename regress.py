#include <bits/stdc++.h>
#include <omp.h>

#include <chrono>
using namespace std;
using namespace chrono;

void initialize(vector<int>& A, int N) {
	for (int i = 0; i < N; i++) {
		A.push_back(rand() % 10);
	}
}

int calculateSum(vector<int>& A, int N) {
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += A[i];
	}
	return sum;
}

vector<int> calculateProduct(vector<int>& A, vector<int>& B, int N) {
	vector<int> productVector(N, 0);
	for (int i = 0; i < N; i++) {
		productVector[i] = A[i] * B[i];
	}
	return productVector;
}

vector<int> calculateSquaredVector(vector<int>& A, int N) {
	vector<int> squaredVector;
	for (int i = 0; i < N; i++) {
		squaredVector.push_back(pow(A[i], 2));
	}
	return squaredVector;
}

int calculateParallelSum(vector<int>& A, int N) {
	int sum = 0;
#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < N; i++) {
		sum += A[i];
	}
	return sum;
}

int main() {
	int N;
	cout << "Define N(no of observations): ";
	cin >> N;

	vector<int> X;
	vector<int> Y;

	initialize(X, N);
	initialize(Y, N);

	vector<int> productVector = calculateProduct(X, Y, N);
	vector<int> squaredVector = calculateSquaredVector(X, N);

	// Serial Approach

	auto serial_start = high_resolution_clock::now();
	int sumOfX = calculateSum(X, N);
	int sumOfY = calculateSum(Y, N);

	int sumOfXSquared = calculateSum(squaredVector, N);
	int sumOfXY = calculateSum(productVector, N);

	// Equation for linear regression
	// y = (alpha) + (beta)x
	float alpha;
	float beta;

	alpha = ((sumOfY * sumOfXSquared) - (sumOfX * sumOfXY)) /
			(sumOfXSquared - pow(sumOfX, 2));

	beta = ((N * sumOfXY) - (sumOfX * sumOfY)) /
		   ((N * sumOfXSquared) - pow(sumOfX, 2));

	auto serial_end = high_resolution_clock::now();
	auto serial_duration =
		duration_cast<milliseconds>(serial_end - serial_start).count();

	cout << "Linear regression equation: " << endl;
	cout << "y = " << alpha << " + " << beta << ".x" << endl;

	cout << "Time taken(serial): " << serial_duration << " milliseconds" << endl
		 << endl;

	// Parallel Approach

	auto parallel_start = high_resolution_clock::now();
	sumOfX = calculateParallelSum(X, N);
	sumOfY = calculateParallelSum(Y, N);

	sumOfXSquared = calculateParallelSum(squaredVector, N);
	sumOfXY = calculateParallelSum(productVector, N);

	// Equation for linear regression
	// y = (alpha) + (beta)x

	alpha = ((sumOfY * sumOfXSquared) - (sumOfX * sumOfXY)) /
			(sumOfXSquared - pow(sumOfX, 2));

	beta = ((N * sumOfXY) - (sumOfX * sumOfY)) /
		   ((N * sumOfXSquared) - pow(sumOfX, 2));

	auto parallel_end = high_resolution_clock::now();
	auto parallel_duration =
		duration_cast<milliseconds>(parallel_end - parallel_start).count();

	cout << "Linear regression equation: " << endl;
	cout << "y = " << alpha << " + " << beta << ".x" << endl;

	cout << "Time taken(parallel): " << parallel_duration << " milliseconds"
		 << endl;
}