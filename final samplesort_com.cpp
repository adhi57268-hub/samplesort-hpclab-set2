#include <mpi.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>

using namespace std;

// Map C++ types to MPI types
template<typename T>
MPI_Datatype MPITypeFrom();
template<> MPI_Datatype MPITypeFrom<int>() { return MPI_INT; }

// Check global sortedness
template <typename T>
bool checkGlobalSortedness(MPI_Comm comm, const vector<T>& localData, int myRank, int p) {
    if (localData.empty()) return true;

    T localFirst = localData.front();
    T localLast  = localData.back();

    MPI_Status status;
    T prevLast;

    bool isSorted = true;
    if (myRank < p - 1) {
        MPI_Send(&localLast, 1, MPITypeFrom<T>(), myRank + 1, 0, comm);
    }
    if (myRank > 0) {
        MPI_Recv(&prevLast, 1, MPITypeFrom<T>(), myRank - 1, 0, comm, &status);
        if (prevLast > localFirst) isSorted = false;
    }

    int localSorted = isSorted ? 1 : 0;
    int globalSorted = 0;
    MPI_Allreduce(&localSorted, &globalSorted, 1, MPI_INT, MPI_MIN, comm);

    return globalSorted == 1;
}

template<class Element>
void parallelSort(MPI_Comm comm, vector<Element>& data,
                  MPI_Datatype mpiType, int p, int myRank)
{
    double sortStart = MPI_Wtime();
    double commTime = 0.0;

    random_device rd;
    mt19937 rndEngine(rd());
    uniform_int_distribution<size_t> dataGen(0, data.size() - 1);

    vector<Element> locS;
    const int a = (int)(16 * log(p) / log(2.0));
    for (size_t i = 0; i < (size_t)(a + 1); ++i)
        locS.push_back(data[dataGen(rndEngine)]);

    vector<Element> s(locS.size() * p);

    double c1 = MPI_Wtime();
    MPI_Allgather(locS.data(), locS.size(), mpiType,
                  s.data(), locS.size(), mpiType, comm);
    double c2 = MPI_Wtime();
    commTime += c2 - c1;

    sort(s.begin(), s.end());

    for (size_t i = 0; i < p - 1; ++i)
        s[i] = s[(a + 1) * (i + 1)];
    s.resize(p - 1);

    vector<vector<Element>> buckets(p);
    for (auto& bucket : buckets)
        bucket.reserve((data.size() / p) * 2);

    for (auto& el : data) {
        const auto bound = upper_bound(s.begin(), s.end(), el);
        buckets[bound - s.begin()].push_back(el);
    }

    data.clear();

    vector<int> sCounts, sDispls, rCounts(p), rDispls(p + 1);
    sDispls.push_back(0);
    for (auto& bucket : buckets) {
        data.insert(data.end(), bucket.begin(), bucket.end());
        sCounts.push_back((int)bucket.size());
        sDispls.push_back((int)bucket.size() + sDispls.back());
    }

    c1 = MPI_Wtime();
    MPI_Alltoall(sCounts.data(), 1, MPI_INT,
                 rCounts.data(), 1, MPI_INT, comm);
    c2 = MPI_Wtime();
    commTime += c2 - c1;

    rDispls[0] = 0;
    for (int i = 1; i <= p; i++)
        rDispls[i] = rCounts[i - 1] + rDispls[i - 1];

    vector<Element> rData(rDispls.back());

    c1 = MPI_Wtime();
    MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(), mpiType,
                  rData.data(), rCounts.data(), rDispls.data(), mpiType, comm);
    c2 = MPI_Wtime();
    commTime += c2 - c1;

    sort(rData.begin(), rData.end());
    rData.swap(data);

    double sortEnd = MPI_Wtime();
    if (myRank == 0) {
        cout << "Sorting phase (excl. comm): " << (sortEnd - sortStart - commTime) << " s" << endl;
        cout << "Communication time: " << commTime << " s" << endl;
    }
}

// Deterministic generator: each rank generates its own chunk
vector<int> generateLocalChunk(uint64_t globalSize, int maxValue, int myRank, int p) {
    uint64_t chunkSize = globalSize / p;
    vector<int> arr(chunkSize);

    // Seed per rank ensures reproducibility
    mt19937_64 gen(myRank + 123456789ULL);
    uniform_int_distribution<int> dist(0, maxValue);

    for (uint64_t i = 0; i < chunkSize; i++) {
        arr[i] = dist(gen);
    }
    return arr;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int p, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    uint64_t globalSize = 1000000; // default 1e6
    int maxValue = 100;

    if (myRank == 0) {
        cout << "Enter total number of elements: ";
        cin >> globalSize;
        cout << "Enter maximum value: ";
        cin >> maxValue;

        if (globalSize % p != 0) {
            cout << "Warning: globalSize not divisible by " << p
                 << ". Truncating to " << (globalSize / p) * p << endl;
            globalSize = (globalSize / p) * p;
        }
    }

    MPI_Bcast(&globalSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double totalStart = MPI_Wtime();

    // Generate only local portion
    vector<int> localData = generateLocalChunk(globalSize, maxValue, myRank, p);

    // Parallel sort
    parallelSort(MPI_COMM_WORLD, localData, MPI_INT, p, myRank);

    // Check global sortedness
    bool globalSorted = checkGlobalSortedness(MPI_COMM_WORLD, localData, myRank, p);

    double totalEnd = MPI_Wtime();

    if (myRank == 0) {
        cout << "Total execution time: " << (totalEnd - totalStart) << " s" << endl;
        cout << "Is globally sorted: " << (globalSorted ? "YES" : "NO") << endl;
    }

    MPI_Finalize();
    return 0;
}
