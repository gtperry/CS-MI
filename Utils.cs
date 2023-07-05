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

        public static void print1d<T>(T[] array)
        {
            Console.WriteLine($"[{string.Join(", ", array)}]");
        }

        public static void print2d<T>(T[,] array)
        {
            Console.WriteLine("[");
            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("{0}", array[i, j]);
                    if (j < array.GetLength(1) - 1)
                    {
                        Console.Write(", ");
                    }
                }
                Console.Write("]");
                if (i < array.GetLength(0) - 1)
                {
                    Console.WriteLine(",");
                }
            }
            Console.WriteLine();
            Console.WriteLine("]");
        }

        public static double[] GenerateRandomNumbers(int length)
        {
            Random rand = new Random();
            double[] numbers = new double[length];

            for (int i = 0; i < length; i++)
            {
                // numbers[i] = rand.NextDouble() * 10;
                // numbers[i] = rand.NextDouble() * 10000; // This works. Seems to be the limit (dependent on GPU memory).
                // numbers[i] = rand.NextDouble() * 100000; // This fails because the numbers are too big and too far apart.
                numbers[i] = rand.NextDouble() * 10 + 1000000; // This now works with bigger numbers, as long as they are close together.
            }

            return numbers;
        }

        /// <summary>
        /// Szudzik pairing function to uniquely encode two natural numbers 
        /// into a single natural number with 100% packing efficiency.
        /// This version only works with positive numbers.
        /// <br/>Source: https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/
        /// </summary>
        /// <param name="x">First number</param>
        /// <param name="y">Second number</param>
        /// <returns></returns>
        public static long szudzikPair(long x, long y)
        {
            return (x >= y ? (x * x) + x + y : (y * y) + x);
        }
    }
}
