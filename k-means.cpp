#include <bits/stdc++.h>
#include <omp.h>

#include <chrono>
using namespace std;
using namespace chrono;

struct Point {
	double x;
	double y;
};

struct Cluster {
	vector<Point> Mean;
	vector<Point> ClusterPoints;
	vector<vector<Point>> Clusters;
};

void printDatapoints(Cluster C, int K, int N) {
	cout << "Data Points are: " << endl;
	cout << "[";
	for (int i = 0; i < N; i++) {
		Point P = C.ClusterPoints[i];
		cout << "(" << P.x << "," << P.y << "), ";
	}
	cout << "]" << endl << endl;
}

void printMean(Cluster C, int K, int N) {
	cout << "Means are: " << endl;
	cout << "[";
	for (int i = 0; i < K; i++) {
		Point P = C.Mean[i];
		cout << "(" << P.x << "," << P.y << "), ";
	}
	cout << "]" << endl << endl;
}

void printCluster(Cluster C, int K, int N) {
	cout << "Clusters are: " << endl;
	for (int i = 0; i < K; i++) {
		Point mean = C.Mean[i];
		cout << "For Mean: (" << mean.x << "," << mean.y << ")" << endl;
		int noOfPoints = C.Clusters[i].size();
		cout << "[";
		for (int j = 0; j < noOfPoints; j++) {
			Point P = C.Clusters[i][j];
			cout << "(" << P.x << "," << P.y << "), ";
		}
		cout << "]" << endl << endl;
	}
}

void print(Cluster C, int K, int N) {
	printDatapoints(C, K, N);
	printMean(C, K, N);
	printCluster(C, K, N);
}

void initialize(vector<double>& A, int N) {
	for (int i = 0; i < N; i++) {
		A.push_back(static_cast<double>(rand()) / RAND_MAX);
	}
}

void initializeCoords(vector<Point>& A, int N) {
	for (int i = 0; i < N; i++) {
		Point P;
		P.x = rand() % 10;
		P.y = rand() % 10;
		A.push_back(P);
	}
}

double calculateDistance(Point A, Point B) {
	int diffX = A.x - B.x;
	int diffY = A.y - B.y;
	int sum = pow(diffX, 2) + pow(diffY, 2);
	double distance = sqrt(sum);
	return distance;
}

void serialKmeans(Cluster& C, int iterationsCount, int K, int N) {
	for (int i = 0; i < iterationsCount; i++) {
		// With each iteration, update mean and clusters
		vector<vector<Point>> temp(K, vector<Point>());
		for (int j = 0; j < N; j++) {
			// Select a point
			Point P = C.ClusterPoints[j];
			double leastDistance = 99.99;
			int index = i;
			for (int x = 0; x < K; x++) {
				// Compare the distance with each mean
				double result = calculateDistance(P, C.Mean[x]);
				if (leastDistance > result) {
					leastDistance = result;
					index = x;
				}
			}
			// Add the point to respective cluster
			temp[index].push_back(P);
		}
		// Update Clusters at the end of iteration
		C.Clusters = temp;
		// Update mean at the end of each iteration
		for (int p = 0; p < K; p++) {
			Point mean;
			double sumX = 0;
			double sumY = 0;
			int noOfPoints = C.Clusters[p].size();
			if (noOfPoints == 0) {
				continue;
			} else {
				for (int q = 0; q < noOfPoints; q++) {
					Point P = C.Clusters[p][q];
					sumX += P.x;
					sumY += P.y;
				}
				mean.x = sumX / noOfPoints;
				mean.y = sumY / noOfPoints;
				C.Mean[p] = mean;
			}
		}
		// cout << "Intermediate ";
		// printMean(C, K, N);
	}
}

void parallelKmeans(Cluster& C, int iterationsCount, int K, int N) {
	for (int i = 0; i < iterationsCount; i++) {
		// With each iteration, update mean and clusters
		vector<vector<Point>> temp(10, vector<Point>());
		for (int j = 0; j < N; j++) {
			// Select a point
			Point P = C.ClusterPoints[j];
			double leastDistance = 99.99;
			int index = i;
			for (int x = 0; x < K; x++) {
				// Compare the distance with each mean
				double result = calculateDistance(P, C.Mean[x]);
				if (leastDistance > result) {
					leastDistance = result;
					index = x;
				}
			}
			// Add the point to respective cluster
			temp[index].push_back(P);
		}
		// Update Clusters at the end of iteration
		C.Clusters = temp;
// Update mean at the end of each iteration
#pragma omp parallel for
		for (int p = 0; p < K; p++) {
			double sumX = 0;
			double sumY = 0;
			int noOfPoints = C.Clusters[p].size();
			for (int q = 0; q < noOfPoints; q++) {
				Point P = C.Clusters[p][q];
				sumX += P.x;
				sumY += P.y;
			}
			Point mean;
			mean.x = sumX / noOfPoints;
			mean.y = sumY / noOfPoints;
			C.Mean[p] = mean;
		}
		// cout << "Intermediate ";
		// printMean(C, K, N);
	}
	// Removing garbage of residual parallel threads
	C.Clusters.resize(K);
}

int main() {
	int N;
	cout << "Define no of points: ";
	cin >> N;

	Cluster C;

	vector<vector<int>> dataset;
	initializeCoords(C.ClusterPoints, N);

	int K;
	cout << "Define K: ";
	cin >> K;

	vector<vector<int>> means;
	initializeCoords(C.Mean, K);

	int iterationsCount;
	cout << "Define no of iterations: ";
	cin >> iterationsCount;

	// cout << "Starting ";
	// printMean(C, K, N);

	auto serial_start = high_resolution_clock::now();
	serialKmeans(C, iterationsCount, K, N);
	auto serial_end = high_resolution_clock::now();
	auto serial_duration =
		duration_cast<milliseconds>(serial_end - serial_start).count();

	cout << "Time taken(serial): " << serial_duration << " milliseconds"
		 << endl;
	// print(C, K, N);

	cout << "_______________________PARALLEL_______________________" << endl
		 << endl;
	// Reset Means for parallel
	C.Mean.clear();
	C.Clusters.clear();
	initializeCoords(C.Mean, K);
	// cout << "Starting ";
	// printMean(C, K, N);
	auto parallel_start = high_resolution_clock::now();
	parallelKmeans(C, iterationsCount, K, N);
	auto parallel_end = high_resolution_clock::now();
	auto parallel_duration =
		duration_cast<milliseconds>(parallel_end - parallel_start).count();

	cout << "Time taken(parallel): " << parallel_duration << " milliseconds"
		 << endl
		 << endl;
	// print(C, K, N);
}