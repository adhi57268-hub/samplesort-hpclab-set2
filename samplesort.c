#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compare_floats(const void *a, const void *b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Total number of elements to sort (example: 10^9)
    long long N = 1000LL;

    // Compute local array size
    long long base = N / size;
    long long rem = N % size;
    long long local_n = base + (rank < rem ? 1 : 0);

    // Allocate local data
    float *local_data = (float*)malloc(local_n * sizeof(float));
    if (!local_data) {
        fprintf(stderr, "Rank %d: Failed to allocate local_data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Seed RNG uniquely per rank
    srand(time(NULL) + rank * 12345);

    // Generate uniform random floats in [0.0,1.0)
    for (long long i = 0; i < local_n; i++) {
        local_data[i] = (float)rand() / (float)RAND_MAX;
    }

    // Local sort
    qsort(local_data, local_n, sizeof(float), compare_floats);

    // Choose local samples (size-1 samples evenly spaced)
    int sample_count = size - 1;
    float *local_samples = (float*)malloc(sample_count * sizeof(float));
    for (int i = 0; i < sample_count; i++) {
        long long idx = (i + 1) * local_n / (sample_count + 1);
        local_samples[i] = local_data[idx];
    }

    // Gather samples at rank 0 using point-to-point
    float *all_samples = NULL;
    if (rank == 0) {
        all_samples = (float*)malloc(sample_count * size * sizeof(float));
    }

    if (rank == 0) {
        // Copy own samples
        for (int i = 0; i < sample_count; i++) {
            all_samples[i] = local_samples[i];
        }
        // Receive samples from other ranks
        for (int src = 1; src < size; src++) {
            MPI_Recv(all_samples + src * sample_count, sample_count, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Send samples to rank 0
        MPI_Send(local_samples, sample_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    free(local_samples);

    // Rank 0 selects pivots by sorting samples and choosing pivots
    float *pivots = (float*)malloc(sample_count * sizeof(float));
    if (rank == 0) {
        qsort(all_samples, sample_count * size, sizeof(float), compare_floats);
        for (int i = 0; i < sample_count; i++) {
            pivots[i] = all_samples[(i + 1) * sample_count];
        }
        free(all_samples);
    }

    // Broadcast pivots using point-to-point sends
    if (rank == 0) {
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(pivots, sample_count, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(pivots, sample_count, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Partition local_data according to pivots
    int *bucket_indices = (int*)malloc((size + 1) * sizeof(int));
    bucket_indices[0] = 0;
    int pidx = 0;
    for (long long i = 0; i < local_n; i++) {
        while (pidx < sample_count && local_data[i] > pivots[pidx]) {
            bucket_indices[++pidx] = i;
        }
    }
    while (pidx < size) {
        bucket_indices[++pidx] = local_n;
    }

    // Send sizes and receive sizes using point-to-point
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *recvcounts = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = bucket_indices[i+1] - bucket_indices[i];
    }

    // Exchange sendcounts: each rank sends its count for destination i to rank i
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            // send to all others
            for (int j = 0; j < size; j++) {
                if (j != rank) {
                    MPI_Send(&sendcounts[j], 1, MPI_INT, j, 2, MPI_COMM_WORLD);
                }
            }
            recvcounts[rank] = sendcounts[rank];
        } else {
            MPI_Recv(&recvcounts[i], 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Calculate displacements
    int *sdispls = (int*)malloc(size * sizeof(int));
    int *rdispls = (int*)malloc(size * sizeof(int));
    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < size; i++) {
        sdispls[i] = sdispls[i-1] + sendcounts[i-1];
        rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }

    int recv_total = 0;
    for (int i = 0; i < size; i++) {
        recv_total += recvcounts[i];
    }

    float *recvbuf = (float*)malloc(recv_total * sizeof(float));

    // Exchange data partitions using non-blocking sends/receives
    MPI_Request *send_reqs = (MPI_Request*)malloc(size * sizeof(MPI_Request));
    MPI_Request *recv_reqs = (MPI_Request*)malloc(size * sizeof(MPI_Request));

    for (int i = 0; i < size; i++) {
        MPI_Irecv(recvbuf + rdispls[i], recvcounts[i], MPI_FLOAT, i, 3, MPI_COMM_WORLD, &recv_reqs[i]);
    }
    for (int i = 0; i < size; i++) {
        MPI_Isend(local_data + bucket_indices[i], sendcounts[i], MPI_FLOAT, i, 3, MPI_COMM_WORLD, &send_reqs[i]);
    }

    MPI_Waitall(size, recv_reqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(size, send_reqs, MPI_STATUSES_IGNORE);

    free(local_data);
    free(bucket_indices);
    free(sendcounts);
    free(recvcounts);
    free(sdispls);
    free(rdispls);
    free(send_reqs);
    free(recv_reqs);
    free(pivots);

    // Sort received data locally
    qsort(recvbuf, recv_total, sizeof(float), compare_floats);

    // Print first 10 elements to verify correctness
    printf("Rank %d: first 10 elements: ", rank);
    for (int i = 0; i < (recv_total < 10 ? recv_total : 10); i++) {
        printf("%.6f ", recvbuf[i]);
    }
    printf("\n");

    free(recvbuf);

    MPI_Finalize();
    return 0;
}
