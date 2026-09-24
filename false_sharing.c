#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define N 17064000LL
#define THREADS 8

static inline uint32_t collatz_steps(uint64_t n)
{
    uint32_t steps = 0;

    while (n > 1)
    {
        if ((n & 1ULL) == 0)
            n >>= 1;
        else
            n = 3 * n + 1;

        steps++;
    }

    return steps;
}

void run_naive(void)
{
    double times[3];

    printf("\n=== Variant 1: Naive hit_count[tid]++ ===\n");

    for (int run = 0; run < 3; run++)
    {
        int hit_count[THREADS] = {0};
        uint32_t max_steps = 0;

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(THREADS) schedule(static) reduction(max:max_steps)
        for (long long i = 1; i <= N; i++)
        {
            uint32_t steps = collatz_steps((uint64_t)i);

            if (steps > max_steps)
                max_steps = steps;

            if (steps > 100)
            {
                int tid = omp_get_thread_num();
                hit_count[tid]++;
            }
        }

        double end = omp_get_wtime();

        long long total_hits = 0;

        for (int i = 0; i < THREADS; i++)
            total_hits += hit_count[i];

        times[run] = end - start;

        printf("Run %d: %.6f s | hits=%lld | max_steps=%u\n",
               run + 1,
               times[run],
               total_hits,
               max_steps);
    }

    double avg = (times[1] + times[2]) / 2.0;

    printf("Naive Avg = %.6f s\n", avg);
    printf("Naive Throughput = %.2f iter/sec\n", N / avg);
}

void run_reduction(void)
{
    double times[3];

    printf("\n=== Variant 2: OpenMP Reduction ===\n");

    for (int run = 0; run < 3; run++)
    {
        long long total_hits = 0;
        uint32_t max_steps = 0;

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(THREADS) schedule(static) \
            reduction(+:total_hits) reduction(max:max_steps)
        for (long long i = 1; i <= N; i++)
        {
            uint32_t steps = collatz_steps((uint64_t)i);

            if (steps > max_steps)
                max_steps = steps;

            if (steps > 100)
                total_hits++;
        }

        double end = omp_get_wtime();

        times[run] = end - start;

        printf("Run %d: %.6f s | hits=%lld | max_steps=%u\n",
               run + 1,
               times[run],
               total_hits,
               max_steps);
    }

    double avg = (times[1] + times[2]) / 2.0;

    printf("Reduction Avg = %.6f s\n", avg);
    printf("Reduction Throughput = %.2f iter/sec\n", N / avg);
}

int main(void)
{
    omp_set_dynamic(0);

    printf("N = %lld\n", N);
    printf("Physical Threads Used = %d\n", THREADS);

    run_naive();
    run_reduction();

    return 0;
}