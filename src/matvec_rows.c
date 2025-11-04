#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void ParseDimensions(int argc, char **argv, int my_rank, int *rows, int *cols);
static void BuildCounts(int total, int comm_sz, int *counts);
static void BuildDisplacements(int comm_sz, const int *counts, int *displs);
static void GenerateMatrix(int my_rank, int rows, int cols, int *matrix);
static void GenerateVector(int my_rank, int cols, int *vector);
static void ScatterMatrixRows(const int *matrix, int *local_matrix, int rows, int cols,
    const int *row_counts, const int *row_displs, const int *scatter_counts,
    const int *scatter_displs, int my_rank);
static void ComputeLocalProduct(int local_rows, int cols, const int *local_matrix,
    const int *vector, int *local_result);
static void GatherResults(const int *local_result, int *global_result, const int *row_counts,
    const int *row_displs, int local_rows);
static void Validate(int my_rank, const int *matrix, const int *vector, const int *result,
    int rows, int cols);

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

    int *row_counts = calloc(comm_sz, sizeof(int));
    int *row_displs = calloc(comm_sz, sizeof(int));
    int *scatter_counts = calloc(comm_sz, sizeof(int));
    int *scatter_displs = calloc(comm_sz, sizeof(int));

    BuildCounts(rows, comm_sz, row_counts);
    BuildDisplacements(comm_sz, row_counts, row_displs);
    for (int i = 0; i < comm_sz; i++) {
        scatter_counts[i] = row_counts[i] * cols;
    }
    BuildDisplacements(comm_sz, scatter_counts, scatter_displs);

    int *matrix = NULL;
    if (my_rank == 0) {
        matrix = calloc(rows * cols, sizeof(int));
    }
    GenerateMatrix(my_rank, rows, cols, matrix);

    int *vector = calloc(cols > 0 ? cols : 1, sizeof(int));
    GenerateVector(my_rank, cols, vector);
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    const int local_rows = row_counts[my_rank];
    const int local_elems = scatter_counts[my_rank];
    int *local_matrix = calloc(local_elems > 0 ? local_elems : 1, sizeof(int));
    ScatterMatrixRows(matrix, local_matrix, rows, cols, row_counts, row_displs, scatter_counts,
        scatter_displs, my_rank);

    int *local_result = calloc(local_rows > 0 ? local_rows : 1, sizeof(int));
    ComputeLocalProduct(local_rows, cols, local_matrix, vector, local_result);

    int *global_result = NULL;
    if (my_rank == 0) {
        global_result = calloc(rows, sizeof(int));
    }
    GatherResults(local_result, global_result, row_counts, row_displs, local_rows);

    Validate(my_rank, matrix, vector, global_result, rows, cols);

    free(global_result);
    free(local_result);
    free(local_matrix);
    free(vector);
    free(matrix);
    free(scatter_displs);
    free(scatter_counts);
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
            printf("Использование: mpirun -np <p> ./matvec_rows <rows> <cols>\n");
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

static void ScatterMatrixRows(const int *matrix, int *local_matrix, int rows, int cols,
    const int *row_counts, const int *row_displs, const int *scatter_counts,
    const int *scatter_displs, int my_rank)
{
    (void)rows;
    (void)cols;
    (void)row_counts;
    (void)row_displs;
    MPI_Scatterv(matrix, scatter_counts, scatter_displs, MPI_INT, local_matrix,
        scatter_counts[my_rank] > 0 ? scatter_counts[my_rank] : 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static void ComputeLocalProduct(int local_rows, int cols, const int *local_matrix,
    const int *vector, int *local_result)
{
    for (int i = 0; i < local_rows; i++) {
        int accum = 0;
        for (int j = 0; j < cols; j++) {
            accum += local_matrix[i * cols + j] * vector[j];
        }
        local_result[i] = accum;
    }
}

static void GatherResults(const int *local_result, int *global_result, const int *row_counts,
    const int *row_displs, int local_rows)
{
    MPI_Gatherv(local_result, local_rows, MPI_INT, global_result, row_counts, row_displs,
        MPI_INT, 0, MPI_COMM_WORLD);
}

static void Validate(int my_rank, const int *matrix, const int *vector, const int *result,
    int rows, int cols)
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
        printf("Построчное умножение проверено для матрицы %d x %d.\n", rows, cols);
    } else {
        printf("Обнаружено несоответствие в построчном умножении.\n");
    }
}
