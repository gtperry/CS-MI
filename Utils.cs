using System;
using System.Diagnostics;

namespace CSMI
{
    public static class Utils
    {
        /// <summary>
        /// Measures the execution time of a function and prints it to the console.
        /// </summary>
        /// <param name="functionName">Name to be printed in the console</param>
        /// <param name="function">Function to be executed and/or measured</param>
        /// <param name="measure">Determines whether the function should be measured or not</param>
        public static T MeasureExecutionTime<T>(
            string functionName,
            Func<T> function,
            bool measure = true,
            bool printOutput = false
        )
        {
            T functionOutput;

            if (!measure)
            {
                functionOutput = function();
            }
            else
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Restart();
                functionOutput = function();
                stopwatch.Stop();
                Console.Write($"Elapsed time for '{functionName}': {stopwatch.ElapsedMilliseconds} ms");
            }

            if (printOutput)
            {
                Console.WriteLine($" -> {functionOutput}");
            }
            else
            {
                Console.WriteLine();
            }

            return functionOutput;
        }

        public static long szudzikPair(long x, long y)
        {
            return (x >= y ? (x * x) + x + y : (y * y) + x);
        }
    }
}
