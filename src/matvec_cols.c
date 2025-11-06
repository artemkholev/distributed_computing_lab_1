#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void ParseDimensions(int argc, char **argv, int my_rank, int *rows, int *cols);
static void BuildCounts(int total, int comm_sz, int *counts);
static void BuildDisplacements(int comm_sz, const int *counts, int *displs);
static void GenerateMatrix(int my_rank, int rows, int cols, int *matrix);
static void GenerateVector(int my_rank, int cols, int *vector);
static void PackColumns(const int *matrix, int rows, int cols, const int *col_counts,
    const int *col_displs, int comm_sz, int *packed);
static void ComputeLocalContribution(int rows, int local_cols, const int *local_matrix,
    const int *local_vector, int *partial_result);
static void ReduceResults(int my_rank, int rows, const int *partial_result, int *final_result);
static void Validate(int my_rank, const int *matrix, const int *vector, const int *result,
    int rows, int cols, double elapsed);

int main(int argc, char **argv)
{
    int my_rank = 0;
    int comm_sz = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

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

    int *col_counts = calloc(comm_sz, sizeof(int));
    int *col_displs = calloc(comm_sz, sizeof(int));
    int *scatter_counts = calloc(comm_sz, sizeof(int));
    int *scatter_displs = calloc(comm_sz, sizeof(int));

    BuildCounts(cols, comm_sz, col_counts);
    BuildDisplacements(comm_sz, col_counts, col_displs);
    for (int i = 0; i < comm_sz; i++) {
        scatter_counts[i] = rows * col_counts[i];
    }
    BuildDisplacements(comm_sz, scatter_counts, scatter_displs);

    int *matrix = NULL;
    int *packed_matrix = NULL;
    if (my_rank == 0) {
        matrix = calloc(rows * cols, sizeof(int));
        packed_matrix = calloc(rows * cols, sizeof(int));
    }
    GenerateMatrix(my_rank, rows, cols, matrix);

    if (my_rank == 0) {
        PackColumns(matrix, rows, cols, col_counts, col_displs, comm_sz, packed_matrix);
    }

    const int local_cols = col_counts[my_rank];
    const int local_size = rows * (local_cols > 0 ? local_cols : 1);
    int *local_matrix = calloc(local_size, sizeof(int));

    int *vector = calloc(cols > 0 ? cols : 1, sizeof(int));
    GenerateVector(my_rank, cols, vector);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatterv(packed_matrix, scatter_counts, scatter_displs, MPI_INT, local_matrix,
        scatter_counts[my_rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_vector = calloc(local_cols > 0 ? local_cols : 1, sizeof(int));
    if (local_cols > 0) {
        for (int j = 0; j < local_cols; j++) {
            const int global_col = col_displs[my_rank] + j;
            local_vector[j] = vector[global_col];
        }
    }

    int *partial_result = calloc(rows > 0 ? rows : 1, sizeof(int));
    ComputeLocalContribution(rows, local_cols, local_matrix, local_vector, partial_result);

    int *final_result = NULL;
    if (my_rank == 0) {
        final_result = calloc(rows, sizeof(int));
    }
    ReduceResults(my_rank, rows, partial_result, final_result);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed = MPI_Wtime() - start;

    Validate(my_rank, matrix, vector, final_result, rows, cols, elapsed);

    free(final_result);
    free(partial_result);
    free(local_vector);
    free(vector);
    free(local_matrix);
    free(packed_matrix);
    free(matrix);
    free(scatter_displs);
    free(scatter_counts);
    free(col_displs);
    free(col_counts);

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
            printf("Использование: mpirun -np <p> ./matvec_cols <rows> <cols>\n");
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

static void BuildCounts(int total, int comm_sz, int *counts)
{
    for (int i = 0; i < comm_sz; i++) {
        counts[i] = total / comm_sz;
        if (i < total % comm_sz) {
            counts[i]++;
        }
    }
}

static void BuildDisplacements(int comm_sz, const int *counts, int *displs)
{
    displs[0] = 0;
    for (int i = 1; i < comm_sz; i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
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

static void PackColumns(const int *matrix, int rows, int cols, const int *col_counts,
    const int *col_displs, int comm_sz, int *packed)
{
    int offset = 0;
    for (int rank = 0; rank < comm_sz; rank++) {
        for (int j = 0; j < col_counts[rank]; j++) {
            const int global_col = col_displs[rank] + j;
            for (int i = 0; i < rows; i++) {
                packed[offset++] = matrix[i * cols + global_col];
            }
        }
    }
}

static void ComputeLocalContribution(int rows, int local_cols, const int *local_matrix,
    const int *local_vector, int *partial_result)
{
    for (int i = 0; i < rows; i++) {
        int accum = 0;
        for (int j = 0; j < local_cols; j++) {
            accum += local_matrix[j * rows + i] * local_vector[j];
        }
        partial_result[i] = accum;
    }
}

static void ReduceResults(int my_rank, int rows, const int *partial_result, int *final_result)
{
    (void)my_rank;
    MPI_Reduce(partial_result, final_result, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
        printf("Постолбцовое умножение проверено для матрицы %d x %d (время: %.6f с).\n", rows,
            cols, elapsed);
    } else {
        printf(
            "Обнаружено несоответствие в постолбцовом умножении (время: %.6f с).\n", elapsed);
    }
}
