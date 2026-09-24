#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#define N 17064000LL
#define MOD 1000000007ULL

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

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    double times[3];

    printf("N = %lld\n", N);
    printf("Threads = %d\n\n", threads);

    for (int run = 0; run < 3; run++)
    {
        uint32_t max_steps = 0;
        unsigned long long sum_steps = 0;

        double start = omp_get_wtime();

        #pragma omp parallel for schedule(static) \
            reduction(max:max_steps) reduction(+:sum_steps)
        for (long long i = 1; i <= N; i++)
        {
            uint32_t steps = collatz_steps((uint64_t)i);

            if (steps > max_steps)
                max_steps = steps;

            sum_steps += steps;
        }

        double end = omp_get_wtime();

        times[run] = end - start;

        unsigned long long checksum = sum_steps % MOD;

        printf("Run %d: %.6f s | max_steps=%u | checksum=%llu\n",
               run + 1,
               times[run],
               max_steps,
               checksum);
    }

    printf("\nRun 1 discarded.\n");
    printf("Avg T_k = %.6f s\n", (times[1] + times[2]) / 2.0);

    return 0;
}