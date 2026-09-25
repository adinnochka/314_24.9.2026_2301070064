import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import java.util.Locale;

public class ForkJoinLab1 {

    static final int ITERATIONS = 100_000_000;

    public static void main(String[] args) {

        if (args.length == 0) {
            System.out.println("Usage: java ForkJoinLab1 <number_of_threads>");
            return;
        }

        int targetThreads = Integer.parseInt(args[0]);

        System.out.println("Threads: " + targetThreads);
        System.out.println("Iterations per thread: " + ITERATIONS);
        System.out.println("Starting CPU-intensive workload...");

        long startTime = System.nanoTime();

        ForkJoinPool customPool = new ForkJoinPool(targetThreads);

        double totalResult;

        try {
            totalResult = customPool.submit(() ->
                IntStream.range(0, targetThreads)
                    .parallel()
                    .mapToDouble(threadId -> {

                        double localSum = 0.0;

                        for (int i = 1; i <= ITERATIONS; i++) {
                            localSum += Math.sqrt(i + threadId);
                        }

                        return localSum;
                    })
                    .sum()
            ).join();

        } finally {
            customPool.shutdown();
        }

        long endTime = System.nanoTime();

        double elapsedSeconds =
            (endTime - startTime) / 1_000_000_000.0;

        System.out.printf(
            Locale.US,
            "THREADS=%d | TIME=%.4f seconds | CHECKSUM=%.4f%n",
            targetThreads,
            elapsedSeconds,
            totalResult
        );
    }
}