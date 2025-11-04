#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    unsigned int state;
} LcgGenerator;

static long ParseTotalSamples(int argc, char **argv, int my_rank);
static void LcgSeed(LcgGenerator *gen, unsigned int seed);
static double LcgNextDouble(LcgGenerator *gen);
static long CountLocalHits(long local_samples, int my_rank);
static void PrintSummary(int my_rank, long total_samples, long global_hits, double elapsed);

int main(int argc, char **argv)
{
    int my_rank = 0;
    int comm_sz = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const long total_samples = ParseTotalSamples(argc, argv, my_rank);
    if (total_samples <= 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Количество выборок должно быть положительным.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    long local_samples = total_samples / comm_sz;
    const long remainder = total_samples % comm_sz;
    if (my_rank < remainder) {
        local_samples++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const long local_hits = CountLocalHits(local_samples, my_rank);

    long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    PrintSummary(my_rank, total_samples, global_hits, elapsed);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static long ParseTotalSamples(int argc, char **argv, int my_rank)
{
    long total_samples = 1000000;
    if (my_rank == 0 && argc >= 2) {
        total_samples = strtol(argv[1], NULL, 10);
    }
    MPI_Bcast(&total_samples, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return total_samples;
}

static void LcgSeed(LcgGenerator *gen, unsigned int seed)
{
    gen->state = seed;
}

static double LcgNextDouble(LcgGenerator *gen)
{
    const unsigned int a = 1103515245u;
    const unsigned int c = 12345u;
    gen->state = a * gen->state + c;
    return (double)gen->state / (double)UINT_MAX;
}

static long CountLocalHits(long local_samples, int my_rank)
{
    LcgGenerator generator;
    LcgSeed(&generator, 1234u + (unsigned int)my_rank);

    long hits = 0;
    for (long i = 0; i < local_samples; i++) {
        const double x = 2.0 * LcgNextDouble(&generator) - 1.0;
        const double y = 2.0 * LcgNextDouble(&generator) - 1.0;
        if ((x * x) + (y * y) <= 1.0) {
            hits++;
        }
    }
    return hits;
}

static void PrintSummary(int my_rank, long total_samples, long global_hits, double elapsed)
{
    if (my_rank != 0) {
        return;
    }

    const double pi_estimate = 4.0 * (double)global_hits / (double)total_samples;
    printf("Метод Монте-Карло для числа пи:\n");
    printf("  выборок: %ld\n", total_samples);
    printf("  оценка_pi: %.8f\n", pi_estimate);
    printf("  время_сек: %.6f\n", elapsed);
}
