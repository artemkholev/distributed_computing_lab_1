# Лабораторная работа №1 — Итоговые инструкции

Все исходные коды для лабораторной находятся в каталоге `src`.

## Подготовка окружения
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update
  sudo apt install -y build-essential openmpi-bin libopenmpi-dev
  ```
- **macOS (Homebrew):**
  ```bash
  brew update
  brew install open-mpi
  ```

## Сборка проекта
```bash
make -C distributed_computing_lab_1              # Собрать все примеры в каталоге build/
make -C distributed_computing_lab_1 clean        # Очистить артефакты сборки
```

## Запуск программ
- **Оценка числа π (задание 1):**
  ```bash
  mpirun -np 4 build/monte_carlo_pi 10000000
  ```
- **Умножение матрицы на вектор:**
  - Разбиение по строкам: `mpirun -np 4 build/matvec_rows 1200 1200`
  - Разбиение по столбцам: `mpirun -np 4 build/matvec_cols 1200 1200`
  - Блочное разбиение (квадрат числа процессов): `mpirun -np 4 build/matvec_blocks 1200 1200`
- **Алгоритм Кэннона (задание 3):**
  ```bash
  mpirun -np 4 build/matmul_cannon 1024
  ```

> **Важно:** для блочного умножения и Кэннона количество процессов должно быть квадратом натурального числа. Размер матриц должен делиться на корень из числа процессов.


