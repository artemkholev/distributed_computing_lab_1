#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void ParseDimensions(int argc, char **argv, int my_rank, int *rows, int *cols);
static void BuildCounts(int total, int parts, int *counts);
static void BuildDisplacements(int parts, const int *counts, int *displs);
static int BuildProcessGrid(int comm_sz);
static void GenerateMatrix(int my_rank, int rows, int cols, int *matrix);
static void GenerateVector(int my_rank, int cols, int *vector);
static void DistributeBlocks(const int *matrix, int *local_matrix, int rows, int cols,
    const int *row_counts, const int *row_displs, const int *col_counts, const int *col_displs,
    int my_rank, int grid_dim, int local_rows, int local_cols);
static void DistributeVectorSegments(const int *vector, int *local_vector, const int *col_counts,
    const int *col_displs, int my_rank, int grid_dim, int local_cols);
static void MultiplyLocalBlock(int local_rows, int local_cols, const int *local_matrix,
    const int *local_vector, int *partial_result);
static void ReduceRowResults(int my_rank, int grid_dim, int local_rows, int *partial_result,
    int *row_result);
static void GatherRowBlocks(int my_rank, int grid_dim, const int *row_counts,
    const int *row_displs, int local_rows, const int *row_result, int *global_result);
static void Validate(int my_rank, const int *matrix, const int *vector, const int *result,
    int rows, int cols, double elapsed);

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
            fprintf(stderr, "Для блочного разбиения число процессов должно быть квадратом целого.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows = 0;
    int cols = 0;
    ParseDimensions(argc, argv, my_rank, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Размеры матрицы должны быть положительными целыми числами.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double start = 0.0;
    double elapsed = 0.0;

    int *row_counts = calloc(grid_dim, sizeof(int));
    int *row_displs = calloc(grid_dim, sizeof(int));
    int *col_counts = calloc(grid_dim, sizeof(int));
    int *col_displs = calloc(grid_dim, sizeof(int));

    BuildCounts(rows, grid_dim, row_counts);
    BuildDisplacements(grid_dim, row_counts, row_displs);
    BuildCounts(cols, grid_dim, col_counts);
    BuildDisplacements(grid_dim, col_counts, col_displs);

    const int grid_row = my_rank / grid_dim;
    const int grid_col = my_rank % grid_dim;
    const int local_rows = row_counts[grid_row];
    const int local_cols = col_counts[grid_col];

    int *matrix = NULL;
    if (my_rank == 0) {
        matrix = calloc(rows * cols, sizeof(int));
    }
    GenerateMatrix(my_rank, rows, cols, matrix);

    int *local_matrix = calloc(local_rows * (local_cols > 0 ? local_cols : 1), sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    DistributeBlocks(matrix, local_matrix, rows, cols, row_counts, row_displs, col_counts,
        col_displs, my_rank, grid_dim, local_rows, local_cols);

    int *vector = NULL;
    if (my_rank == 0) {
        vector = calloc(cols, sizeof(int));
    }
    GenerateVector(my_rank, cols, vector);

    int *local_vector = calloc(local_cols > 0 ? local_cols : 1, sizeof(int));
    DistributeVectorSegments(vector, local_vector, col_counts, col_displs, my_rank, grid_dim,
        local_cols);

    int *partial_result = calloc(local_rows > 0 ? local_rows : 1, sizeof(int));
    MultiplyLocalBlock(local_rows, local_cols, local_matrix, local_vector, partial_result);

    int *row_result = calloc(local_rows > 0 ? local_rows : 1, sizeof(int));
    ReduceRowResults(my_rank, grid_dim, local_rows, partial_result, row_result);

    int *global_result = NULL;
    if (my_rank == 0) {
        global_result = calloc(rows, sizeof(int));
    }
    GatherRowBlocks(my_rank, grid_dim, row_counts, row_displs, local_rows, row_result,
        global_result);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed = MPI_Wtime() - start;

    Validate(my_rank, matrix, vector, global_result, rows, cols, elapsed);

    free(global_result);
    free(row_result);
    free(partial_result);
    free(local_vector);
    free(vector);
    free(local_matrix);
    free(matrix);
    free(col_displs);
    free(col_counts);
    free(row_displs);
    free(row_counts);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void ParseDimensions(int argc, char **argv, int my_rank, int *rows, int *cols)
{
    if (my_rank == 0) {
        if (argc >= 3) {
            *rows = atoi(argv[1]);
            *cols = atoi(argv[2]);
        } else {
            printf("Использование: mpirun -np <q^2> ./matvec_blocks <rows> <cols>\n");
            printf("Введите размеры матрицы: ");
            fflush(stdout);
            if (scanf("%d %d", rows, cols) != 2) {
                *rows = 0;
                *cols = 0;
            }
        }
    }
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static void BuildCounts(int total, int parts, int *counts)
{
    for (int i = 0; i < parts; i++) {
        counts[i] = total / parts;
        if (i < total % parts) {
            counts[i]++;
        }
    }
}

static void BuildDisplacements(int parts, const int *counts, int *displs)
{
    displs[0] = 0;
    for (int i = 1; i < parts; i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
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

static void GenerateMatrix(int my_rank, int rows, int cols, int *matrix)
{
    if (my_rank != 0) {
        return;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (i + 1) * 10 + (j + 1);
        }
    }
}

static void GenerateVector(int my_rank, int cols, int *vector)
{
    if (my_rank != 0) {
        return;
    }
    for (int j = 0; j < cols; j++) {
        vector[j] = (j % 7) + 1;
    }
}

static void DistributeBlocks(const int *matrix, int *local_matrix, int rows, int cols,
    const int *row_counts, const int *row_displs, const int *col_counts, const int *col_displs,
    int my_rank, int grid_dim, int local_rows, int local_cols)
{
    if (my_rank == 0) {
        for (int rank = 0; rank < grid_dim * grid_dim; rank++) {
            const int target_row = rank / grid_dim;
            const int target_col = rank % grid_dim;
            const int block_rows = row_counts[target_row];
            const int block_cols = col_counts[target_col];
            const int block_row_offset = row_displs[target_row];
            const int block_col_offset = col_displs[target_col];

            if (block_rows == 0 || block_cols == 0) {
                continue;
            }

            if (rank == 0) {
                for (int i = 0; i < block_rows; i++) {
                    for (int j = 0; j < block_cols; j++) {
                        local_matrix[i * block_cols + j] =
                            matrix[(block_row_offset + i) * cols + block_col_offset + j];
                    }
                }
            } else {
                int *buffer = calloc(block_rows * block_cols, sizeof(int));
                for (int i = 0; i < block_rows; i++) {
                    for (int j = 0; j < block_cols; j++) {
                        buffer[i * block_cols + j] =
                            matrix[(block_row_offset + i) * cols + block_col_offset + j];
                    }
                }
                MPI_Send(buffer, block_rows * block_cols, MPI_INT, rank, 0, MPI_COMM_WORLD);
                free(buffer);
            }
        }
    } else if (local_rows > 0 && local_cols > 0) {
        MPI_Recv(local_matrix, local_rows * local_cols, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }

    (void)rows;
    (void)cols;
    (void)row_counts;
    (void)row_displs;
    (void)col_counts;
    (void)col_displs;
}

static void DistributeVectorSegments(const int *vector, int *local_vector, const int *col_counts,
    const int *col_displs, int my_rank, int grid_dim, int local_cols)
{
    if (my_rank == 0) {
        for (int rank = 0; rank < grid_dim * grid_dim; rank++) {
            const int target_col = rank % grid_dim;
            const int segment_cols = col_counts[target_col];
            const int segment_offset = col_displs[target_col];
            if (segment_cols == 0) {
                continue;
            }
            if (rank == 0) {
                for (int j = 0; j < segment_cols; j++) {
                    local_vector[j] = vector[segment_offset + j];
                }
            } else {
                MPI_Send(vector + segment_offset, segment_cols, MPI_INT, rank, 1, MPI_COMM_WORLD);
            }
        }
    } else if (local_cols > 0) {
        MPI_Recv(local_vector, local_cols, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    (void)col_counts;
    (void)col_displs;
}

static void MultiplyLocalBlock(int local_rows, int local_cols, const int *local_matrix,
    const int *local_vector, int *partial_result)
{
    for (int i = 0; i < local_rows; i++) {
        int accum = 0;
        for (int j = 0; j < local_cols; j++) {
            accum += local_matrix[i * local_cols + j] * local_vector[j];
        }
        partial_result[i] = accum;
    }
}

static void ReduceRowResults(int my_rank, int grid_dim, int local_rows, int *partial_result,
    int *row_result)
{
    const int grid_row = my_rank / grid_dim;
    const int grid_col = my_rank % grid_dim;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, grid_row, grid_col, &row_comm);

    MPI_Reduce(partial_result, row_result, local_rows, MPI_INT, MPI_SUM, 0, row_comm);

    MPI_Comm_free(&row_comm);
    (void)grid_col;
}

static void GatherRowBlocks(int my_rank, int grid_dim, const int *row_counts,
    const int *row_displs, int local_rows, const int *row_result, int *global_result)
{
    const int grid_col = my_rank % grid_dim;

    if (grid_col == 0 && my_rank != 0 && local_rows > 0) {
        MPI_Send(row_result, local_rows, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        if (local_rows > 0) {
            for (int i = 0; i < local_rows; i++) {
                global_result[i] = row_result[i];
            }
        }
        for (int row = 1; row < grid_dim; row++) {
            const int sender_rank = row * grid_dim;
            const int rows_to_copy = row_counts[row];
            if (rows_to_copy == 0) {
                continue;
            }
            MPI_Recv(global_result + row_displs[row], rows_to_copy, MPI_INT, sender_rank, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

static void Validate(int my_rank, const int *matrix, const int *vector, const int *result,
    int rows, int cols, double elapsed)
{
    if (my_rank != 0) {
        return;
    }
    bool ok = true;
    for (int i = 0; i < rows; i++) {
        int accum = 0;
        for (int j = 0; j < cols; j++) {
            accum += matrix[i * cols + j] * vector[j];
        }
        if (accum != result[i]) {
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Блочное умножение проверено для матрицы %d x %d (время: %.6f с).\n", rows, cols,
            elapsed);
    } else {
        printf("Обнаружено несоответствие при блочном умножении (время: %.6f с).\n", elapsed);
    }
}
