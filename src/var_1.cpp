#include <iostream>
#include <mpi.h>

// Дихотомический барьер
void dichotomy_barrier(int rank, int size) {
    int dummy = 1;

    // Сбор снизу вверх
    for (int step = 1; step < size; step <<= 1) {
        int partner = rank ^ step;
        if (partner < size && partner > rank) {
            MPI_Recv(&dummy, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (partner < size && partner < rank) {
            MPI_Send(&dummy, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        }
    }

    // Оповещение сверху вниз
    for (int step = 1; step < size; step <<= 1) {
        int partner = rank ^ step;
        if (partner < size && partner > rank) {
            MPI_Send(&dummy, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
        } else if (partner < size && partner < rank) {
            MPI_Recv(&dummy, 1, MPI_INT, partner, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

// Определение среднего времени одного вызова
double measure_barrier(int rank, int size, int iterations, bool use_native) {
    // Прогрев
    if (use_native) MPI_Barrier(MPI_COMM_WORLD);
    else dichotomy_barrier(rank, size);

    // Синхронизация перед замером
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int i = 0; i < iterations; ++i) {
        if (use_native) MPI_Barrier(MPI_COMM_WORLD);
        else dichotomy_barrier(rank, size);
    }

    // Синхронизация после замеров
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // Получаем время барьера
    double local_elapsed = (t_end - t_start) / iterations;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    return max_elapsed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Число итераций для усреднения
    const int ITERATIONS = 1000;

    if (rank == 0) {
        std::cout << "=== Тест дихотомического барьера MPI ===\n";
        std::cout << "Число процессов: " << size << "\n";
        std::cout << "Число итераций:  " << ITERATIONS << "\n\n";
    }

    // Замер MPI_Barrier()
    double t_native = measure_barrier(rank, size, ITERATIONS, true);

    // Замер дихотомического барьера
    double t_dich = measure_barrier(rank, size, ITERATIONS, false);

    if (rank == 0) {
        std::cout << "--- Результаты ---\n";
        std::cout << "MPI_Barrier()         : "
                  << t_native * 1e6 << " мкс/итерацию\n";
        std::cout << "Дихотомический барьер : "
                  << t_dich   * 1e6 << " мкс/итерацию\n";
        std::cout << "Отношение (своя/встр.): "
                  << t_dich / t_native << "\n";
    }

    MPI_Finalize();
    return 0;
}
