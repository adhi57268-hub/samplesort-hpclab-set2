#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>

// Check if local array is sorted
bool isLocalSorted(const std::vector<long long> &arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i-1] > arr[i]) return false;
    }
    return true;
}

// Global correctness check
bool verifyGlobalSort(const std::vector<long long> &local, int rank, int size, MPI_Comm comm) {
    // 1. Check local sortedness
    int localFlag = isLocalSorted(local) ? 1 : 0;

    // Gather flags at root
    int globalFlag = 1;
    if (rank == 0) {
        globalFlag = localFlag;
        for (int r = 1; r < size; r++) {
            int recvFlag;
            MPI_Recv(&recvFlag, 1, MPI_INT, r, 10, comm, MPI_STATUS_IGNORE);
            if (recvFlag == 0) globalFlag = 0;
        }
        // send result back
        for (int r = 1; r < size; r++) {
            MPI_Send(&globalFlag, 1, MPI_INT, r, 11, comm);
        }
    } else {
        MPI_Send(&localFlag, 1, MPI_INT, 0, 10, comm);
        MPI_Recv(&globalFlag, 1, MPI_INT, 0, 11, comm, MPI_STATUS_IGNORE);
    }

    if (globalFlag == 0) return false;

    // 2. Check boundaries between ranks
    if (rank < size-1 && !local.empty()) {
        long long myLast = local.back();
        MPI_Send(&myLast, 1, MPI_LONG_LONG, rank+1, 20, comm);
    }
    if (rank > 0 && !local.empty()) {
        long long prevLast;
        MPI_Recv(&prevLast, 1, MPI_LONG_LONG, rank-1, 20, comm, MPI_STATUS_IGNORE);
        if (prevLast > local.front()) return false;
    }

    return true;
}


// Parallel sample sort function
void parallelSampleSort(long long n, int rank, int size, MPI_Comm comm, std::vector<long long> &localData) {
    // Step 1: distribute responsibility for generating data
    long long local_n = n / size;
    if (rank < n % size) local_n++;

    localData.resize(local_n);
    srand(time(NULL) + rank); // different seed per rank
    for (long long i = 0; i < local_n; i++) {
        localData[i] = rand() % 1000000000LL;
    }

    // Step 2: local sort
    std::sort(localData.begin(), localData.end());

    // Step 3: choose samples
    int s = size - 1;
    std::vector<long long> samples;
    if (!localData.empty()) {
        for (int i = 1; i <= s; i++) {
            samples.push_back(localData[local_n * i / (s+1)]);
        }
    }

    // Gather samples to root (rank 0) manually (no collectives)
    std::vector<long long> allSamples;
    if (rank == 0) {
        allSamples.resize(s * size);
        int idx = 0;
        for (int r = 0; r < size; r++) {
            if (r == 0) {
                for (auto v : samples) allSamples[idx++] = v;
            } else {
                int recvCount = s;
                MPI_Recv(allSamples.data()+idx, recvCount, MPI_LONG_LONG, r, 1, comm, MPI_STATUS_IGNORE);
                idx += recvCount;
            }
        }
    } else {
        MPI_Send(samples.data(), s, MPI_LONG_LONG, 0, 1, comm);
    }

    // Step 4: choose global pivots
    std::vector<long long> pivots(s);
    if (rank == 0) {
        std::sort(allSamples.begin(), allSamples.end());
        for (int i = 1; i <= s; i++) {
            pivots[i-1] = allSamples[allSamples.size() * i / (s+1)];
        }
        // broadcast pivots
        for (int r = 1; r < size; r++) {
            MPI_Send(pivots.data(), s, MPI_LONG_LONG, r, 2, comm);
        }
    } else {
        MPI_Recv(pivots.data(), s, MPI_LONG_LONG, 0, 2, comm, MPI_STATUS_IGNORE);
    }

    // Step 5: partition local data
    std::vector<std::vector<long long>> buckets(size);
    int idx = 0;
    for (auto v : localData) {
        while (idx < s && v > pivots[idx]) idx++;
        buckets[idx].push_back(v);
    }

    // Step 6: exchange buckets
    std::vector<long long> recvData;
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        // send count then data
        long long count = buckets[r].size();
        MPI_Send(&count, 1, MPI_LONG_LONG, r, 3, comm);
        if (count > 0) MPI_Send(buckets[r].data(), count, MPI_LONG_LONG, r, 4, comm);
    }

    // also keep own bucket
    recvData.insert(recvData.end(), buckets[rank].begin(), buckets[rank].end());

    // receive from others
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        long long count;
        MPI_Recv(&count, 1, MPI_LONG_LONG, r, 3, comm, MPI_STATUS_IGNORE);
        if (count > 0) {
            std::vector<long long> temp(count);
            MPI_Recv(temp.data(), count, MPI_LONG_LONG, r, 4, comm, MPI_STATUS_IGNORE);
            recvData.insert(recvData.end(), temp.begin(), temp.end());
        }
    }

    // Step 7: final local sort
    std::sort(recvData.begin(), recvData.end());
    localData.swap(recvData);
}

// Scaling experiment
void testScaling(int rank, int size, MPI_Comm comm) {
    std::vector<long long> localData;
    std::vector<int> testSizes = {100000}; // 10^6 to 10^8 (adjustable)
    // 10^10 is possible but very large for demo, test with smaller then scale up
    for (auto n : testSizes) {
        double start = MPI_Wtime();
        parallelSampleSort(n, rank, size, comm, localData);
        double end = MPI_Wtime();

        bool correct = verifyGlobalSort(localData, rank, size, comm);
        if (rank == 0) {
            std::cout << "n=" << n << ", time=" << (end-start) 
                      << "s, correct=" << (correct ? "✅" : "❌") << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    testScaling(rank, size, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
