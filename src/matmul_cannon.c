#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int BuildProcessGrid(int comm_sz);
static int ParseMatrixSize(int argc, char **argv, int my_rank);
static void GenerateMatrix(int my_rank, int size, int *matrix, int seed_offset);
static void DistributeBlocks(const int *matrix, int *local_block, int size, int block_dim,
    int my_rank, int grid_dim);
static void MultiplyLocal(int block_dim, const int *a, const int *b, int *c);
static void GatherBlocks(const int *local_block, int size, int block_dim, int my_rank,
    int grid_dim, int *result);
static void Validate(int my_rank, int size, const int *a, const int *b, const int *c,
    double elapsed);

int main(int argc, char **argv)
{
    int my_rank = 0;
    int comm_sz = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const int grid_dim = BuildProcessGrid(comm_sz);
    if (grid_dim == 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Для алгоритма Кэннона число процессов должно быть квадратом целого.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int size = ParseMatrixSize(argc, argv, my_rank);
    if (size <= 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Размер матрицы должен быть положительным целым числом.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (size % grid_dim != 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Размер матрицы должен делиться на sqrt(p) для алгоритма Кэннона.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int block_dim = size / grid_dim;
    const int block_elems = block_dim * block_dim;

    double start = 0.0;
    double elapsed = 0.0;

    int *matrix_a = NULL;
    int *matrix_b = NULL;
    if (my_rank == 0) {
        matrix_a = calloc(size * size, sizeof(int));
        matrix_b = calloc(size * size, sizeof(int));
    }
    GenerateMatrix(my_rank, size, matrix_a, 0);
    GenerateMatrix(my_rank, size, matrix_b, 100);

    int *local_a = calloc(block_elems, sizeof(int));
    int *local_b = calloc(block_elems, sizeof(int));
    int *local_c = calloc(block_elems, sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    DistributeBlocks(matrix_a, local_a, size, block_dim, my_rank, grid_dim);
    DistributeBlocks(matrix_b, local_b, size, block_dim, my_rank, grid_dim);

    int dims[2] = {grid_dim, grid_dim};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2] = {0, 0};
    MPI_Cart_coords(grid_comm, my_rank, 2, coords);

    int left = 0;
    int right = 0;
    int up = 0;
    int down = 0;
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    for (int i = 0; i < coords[0]; i++) {
        MPI_Sendrecv_replace(local_a, block_elems, MPI_INT, left, 0, right, 0, grid_comm,
            MPI_STATUS_IGNORE);
    }
    for (int j = 0; j < coords[1]; j++) {
        MPI_Sendrecv_replace(local_b, block_elems, MPI_INT, up, 1, down, 1, grid_comm,
            MPI_STATUS_IGNORE);
    }

    for (int stage = 0; stage < grid_dim; stage++) {
        MultiplyLocal(block_dim, local_a, local_b, local_c);

        if (stage < grid_dim - 1) {
            MPI_Sendrecv_replace(local_a, block_elems, MPI_INT, left, 2, right, 2, grid_comm,
                MPI_STATUS_IGNORE);
            MPI_Sendrecv_replace(local_b, block_elems, MPI_INT, up, 3, down, 3, grid_comm,
                MPI_STATUS_IGNORE);
        }
    }

    MPI_Comm_free(&grid_comm);

    int *matrix_c = NULL;
    if (my_rank == 0) {
        matrix_c = calloc(size * size, sizeof(int));
    }
    GatherBlocks(local_c, size, block_dim, my_rank, grid_dim, matrix_c);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed = MPI_Wtime() - start;

    Validate(my_rank, size, matrix_a, matrix_b, matrix_c, elapsed);

    free(matrix_c);
    free(local_c);
    free(local_b);
    free(local_a);
    free(matrix_b);
    free(matrix_a);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static int BuildProcessGrid(int comm_sz)
{
    int grid_dim = 0;
    for (int candidate = 1; candidate * candidate <= comm_sz; candidate++) {
        if (candidate * candidate == comm_sz) {
            grid_dim = candidate;
        }
    }
    return grid_dim;
}

static int ParseMatrixSize(int argc, char **argv, int my_rank)
{
    int size = 0;
    if (my_rank == 0) {
        if (argc >= 2) {
            size = atoi(argv[1]);
        } else {
            printf("Использование: mpirun -np <q^2> ./matmul_cannon <size>\n");
            printf("Введите размер матрицы: ");
            fflush(stdout);
            if (scanf("%d", &size) != 1) {
                size = 0;
            }
        }
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return size;
}

static void GenerateMatrix(int my_rank, int size, int *matrix, int seed_offset)
{
    if (my_rank != 0) {
        return;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            const int base = (i + seed_offset + 1) % 17;
            matrix[i * size + j] = base * 10 + ((j + seed_offset + 1) % 13);
        }
    }
}

static void DistributeBlocks(const int *matrix, int *local_block, int size, int block_dim,
    int my_rank, int grid_dim)
{
    if (my_rank == 0) {
        for (int rank = 0; rank < grid_dim * grid_dim; rank++) {
            const int row = rank / grid_dim;
            const int col = rank % grid_dim;
            const int row_offset = row * block_dim;
            const int col_offset = col * block_dim;
            if (rank == 0) {
                for (int i = 0; i < block_dim; i++) {
                    memcpy(local_block + i * block_dim,
                        matrix + (row_offset + i) * size + col_offset, block_dim * sizeof(int));
                }
            } else {
                int *buffer = calloc(block_dim * block_dim, sizeof(int));
                for (int i = 0; i < block_dim; i++) {
                    memcpy(buffer + i * block_dim,
                        matrix + (row_offset + i) * size + col_offset, block_dim * sizeof(int));
                }
                MPI_Send(buffer, block_dim * block_dim, MPI_INT, rank, 10, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    } else {
        MPI_Recv(local_block, block_dim * block_dim, MPI_INT, 0, 10, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }
}

static void MultiplyLocal(int block_dim, const int *a, const int *b, int *c)
{
    for (int i = 0; i < block_dim; i++) {
        for (int j = 0; j < block_dim; j++) {
            int accum = c[i * block_dim + j];
            for (int k = 0; k < block_dim; k++) {
                accum += a[i * block_dim + k] * b[k * block_dim + j];
            }
            c[i * block_dim + j] = accum;
        }
    }
}

static void GatherBlocks(const int *local_block, int size, int block_dim, int my_rank,
    int grid_dim, int *result)
{
    if (my_rank == 0) {
        for (int i = 0; i < block_dim; i++) {
            memcpy(result + i * size, local_block + i * block_dim, block_dim * sizeof(int));
        }
        for (int rank = 1; rank < grid_dim * grid_dim; rank++) {
            int *buffer = calloc(block_dim * block_dim, sizeof(int));
            MPI_Recv(buffer, block_dim * block_dim, MPI_INT, rank, 11, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            const int row = rank / grid_dim;
            const int col = rank % grid_dim;
            const int row_offset = row * block_dim;
            const int col_offset = col * block_dim;
            for (int i = 0; i < block_dim; i++) {
                memcpy(result + (row_offset + i) * size + col_offset, buffer + i * block_dim,
                    block_dim * sizeof(int));
            }
            free(buffer);
        }
    } else {
        MPI_Send(local_block, block_dim * block_dim, MPI_INT, 0, 11, MPI_COMM_WORLD);
    }
}

static void Validate(int my_rank, int size, const int *a, const int *b, const int *c,
    double elapsed)
{
    if (my_rank != 0) {
        return;
    }
    bool ok = true;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            long accum = 0;
            for (int k = 0; k < size; k++) {
                accum += (long)a[i * size + k] * (long)b[k * size + j];
            }
            if (accum != c[i * size + j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
    }
    if (ok) {
        printf("Алгоритм Кэннона проверен для матриц %d x %d (время: %.6f с).\n", size, size,
            elapsed);
    } else {
        printf("Обнаружено несоответствие при алгоритме Кэннона (время: %.6f с).\n", elapsed);
    }
}
