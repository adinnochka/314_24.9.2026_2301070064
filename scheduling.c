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

double test_schedule(omp_sched_t kind, int chunk)
{
    double times[3];

    omp_set_schedule(kind, chunk);

    for (int run = 0; run < 3; run++)
    {
        long long total_hits = 0;
        uint32_t max_steps = 0;

        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(THREADS) schedule(runtime) \
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

        printf("  Run %d = %.6f s | hits=%lld | max_steps=%u\n",
               run + 1, times[run], total_hits, max_steps);
    }

    return (times[1] + times[2]) / 2.0;
}

int main(void)
{
    omp_set_dynamic(0);

    double avg;

    printf("\n=== schedule(static) ===\n");
    avg = test_schedule(omp_sched_static, 0);
    printf("AVG = %.6f s\n", avg);

    printf("\n=== schedule(static, 1000) ===\n");
    avg = test_schedule(omp_sched_static, 1000);
    printf("AVG = %.6f s\n", avg);

    printf("\n=== schedule(dynamic, 100) ===\n");
    avg = test_schedule(omp_sched_dynamic, 100);
    printf("AVG = %.6f s\n", avg);

    printf("\n=== schedule(dynamic, 10000) ===\n");
    avg = test_schedule(omp_sched_dynamic, 10000);
    printf("AVG = %.6f s\n", avg);

    printf("\n=== schedule(guided) ===\n");
    avg = test_schedule(omp_sched_guided, 0);
    printf("AVG = %.6f s\n", avg);

    return 0;
}