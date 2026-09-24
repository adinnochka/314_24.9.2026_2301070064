#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define N 17064000ULL
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

int main(void)
{
    double times[3];

    printf("N = %llu\n", (unsigned long long)N);

    for (int run = 0; run < 3; run++)
    {
        uint32_t max_steps = 0;
        uint64_t max_number = 0;
        uint64_t checksum = 0;

        double start = omp_get_wtime();

        for (uint64_t i = 1; i <= N; i++)
        {
            uint32_t steps = collatz_steps(i);

            if (steps > max_steps)
            {
                max_steps = steps;
                max_number = i;
            }

            checksum = (checksum + steps) % MOD;
        }

        double end = omp_get_wtime();

        times[run] = end - start;

        printf("Run %d: %.6f s | max_steps=%u | max_number=%llu | checksum=%llu\n",
               run + 1,
               times[run],
               max_steps,
               (unsigned long long)max_number,
               (unsigned long long)checksum);
    }

    double T_seq = (times[1] + times[2]) / 2.0;

    printf("\nRun 1 discarded.\n");
    printf("T_seq = %.6f s\n", T_seq);

    return 0;
}