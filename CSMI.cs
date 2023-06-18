using System;
using System.Collections.Generic;

using ILGPU;
using ILGPU.Runtime;

namespace CSMI
{
    public class Constants
    {
        public const double LOG_BASE = 2.0;
    }

    // TODO: Rename this to something better
    public class ILGPUInitializer : IDisposable
    {
        public Device dev;
        public Context context;

        public ILGPUInitializer()
        {
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
            this.dev = this.context.GetPreferredDevice(preferCPU: false);
            // If we are not using the CPU, then prefer Cuda over OpenCL for the GPU if both are present
            if (this.dev.AcceleratorType != AcceleratorType.CPU)
            {
                foreach (Device device in this.context)
                {
                    if (device.AcceleratorType == AcceleratorType.Cuda)
                    {
                        this.dev = device;
                        break;
                    }
                }
            }
            Console.WriteLine($"Selected device: {this.dev}");
            Console.WriteLine(new string('=', 10));
        }

        public void Dispose()
        {
            this.context.Dispose();
        }
    }

    public class CSMI : IDisposable
    {
        public ILGPUInitializer ILGPUInitializer { get; private set; }
        public Entropy entropy { get; private set; }
        public MutualInformation mi { get; private set; }

        public CSMI()
        {
            ILGPUInitializer = new ILGPUInitializer();
            entropy = new Entropy(ILGPUInitializer);
            mi = new MutualInformation(ILGPUInitializer);
        }

        // Remember to instantiate MI class with a `using` statement. Called automatically.
        public void Dispose()
        {
            ILGPUInitializer.Dispose();
        }

        double[] GenerateRandomNumbers(int length)
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
        /// This function runs tests against the Java implementation of this library to ensure that the results are reproducible.
        /// </summary>
        void testReproducible()
        {
            // Fixed data for reproducible results
            double[] a = new[] { 4.2, 5.43, 3.221, 7.34235, 1.931, 1.2, 5.43, 8.0, 7.34235, 1.931 };
            // double[] a = new[] { 4.2, -5.43, -3.221, -7.34235, -1.931, -1.2, -5.43, -8.0, -7.34235, -1.931 };
            double[] b = new[] { 2.2, 3.43, 1.221, 9.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 34.931 };
            // double[] a = new[] { 1.2, 2.4, 1.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1 };
            // double[] b = new[] { 2.2, 2.4, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1 };
            double[] c = new[] { 2.2, 3.43, 2.221, 2.34235, 3.931, 3.2, 4.43, 7.0, 7.34235, 34.931 };

            // Test with bigger values
            int orderOfMagnitude = 5;
            int numToMultiply = (int)Math.Pow(10, orderOfMagnitude);
            for (int i = 0; i < a.Length; i++)
            {
                // Addition doesn't change the results, but multiplication does.
                a[i] += numToMultiply;
                b[i] += numToMultiply;
                c[i] += numToMultiply;
            }
            Console.WriteLine($"a: [{string.Join(", ", a)}]");

            // These are the results obtained from the Java implementation of this library for each function.
            var javaResultsMap = new Dictionary<int, (string name, double javaResult)>()
            {
                { 0, ("Entropy", 2.4464393446710155) },
                { 1, ("ConditionalEntropy", 0.6) },
                { 2, ("JointEntropy", 3.121928094887362) },
                { 3, ("MutualInformation", 1.8464393446710157) },
                { 4, ("ConditionalMutualInformation", 0.7509775004326935) },
            };
            // csharpier-ignore-start
            var resultsList = new double[]{
                Utils.MeasureExecutionTime(javaResultsMap[0].name, () => entropy.calculateEntropy(a), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[1].name, () => entropy.calculateConditionalEntropy(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[2].name, () => entropy.calculateJointEntropy(a, b), printOutput: true),
                // Utils.MeasureExecutionTime(javaResultsMap[2].name, () => entropy.calculateJointEntropy2(a, b), printOutput: true),
                // Utils.MeasureExecutionTime(javaResultsMap[2].name, () => entropy.calculateJointEntropy3(a, b), printOutput: true), // TODO: finish or remove
                Utils.MeasureExecutionTime(javaResultsMap[3].name, () => mi.calculateMutualInformation(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[4].name, () => mi.calculateConditionalMutualInformation(a, b, c), printOutput: true),
            };
            // csharpier-ignore-end

            // See if the functions for the C# vs Java implementation are the same
            var decimalPlaces = 12;
            Console.WriteLine(
                $"\nJava vs C# return values test results, accurate to '{decimalPlaces}' decimal places:"
            );
            for (int i = 0; i < resultsList.Length; i++)
            {
                var csharpResult = Math.Round(resultsList[i], decimalPlaces);
                var javaResult = Math.Round(javaResultsMap[i].javaResult, decimalPlaces);
                var testOutcome = (csharpResult == javaResult) ? "PASS" : "FAIL";
                Console.WriteLine($"{i}) {javaResultsMap[i].name}: {testOutcome}");
            }

            Console.WriteLine();
            Console.WriteLine("----------------------------");
            Console.WriteLine();
        }

        void testRandom(int length)
        {
            Console.WriteLine($"LENGTH = {length:n0}");
            double[] a = GenerateRandomNumbers(length);
            double[] b = GenerateRandomNumbers(length);
            double[] c = GenerateRandomNumbers(length);
            // csharpier-ignore-start
            Utils.MeasureExecutionTime("Calculate Entropy", () => entropy.calculateEntropy(a));
            Utils.MeasureExecutionTime("Calculate Conditional Entropy", () => entropy.calculateConditionalEntropy(a, b));
            Utils.MeasureExecutionTime("Calculate Joint Entropy", () => entropy.calculateJointEntropy(a, b));
            // Utils.MeasureExecutionTime("Calculate Joint Entropy 2", () => entropy.calculateJointEntropy2(a, b));
            // Utils.MeasureExecutionTime("Calculate Joint Entropy 3", () => entropy.calculateJointEntropy3(a, b)); // TODO: finish or remove
            Utils.MeasureExecutionTime("Mutual Information", () => mi.calculateMutualInformation(a, b));
            Utils.MeasureExecutionTime("Conditional Mutual Information", () => mi.calculateConditionalMutualInformation(a, b, c));
            // csharpier-ignore-end

            Console.WriteLine();
            Console.WriteLine("----------------------------");
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            #region Command line usage
            Console.WriteLine("Usage: CSMI.exe [OPTION]");
            Console.WriteLine("Options:");
            Console.WriteLine("0 (or none specified) -> Test Reproducible");
            Console.WriteLine("1 -> Test Random");
            Console.WriteLine("WARNING: This library will not work with negative numbers");
            Console.WriteLine(new string('=', 10));
            Console.WriteLine($"Args: {string.Join(" ", args)}");
            int option_selected = 0;
            if (args.Length != 0)
            {
                try
                {
                    option_selected = int.Parse(args[0]);
                }
                catch { }
            }
            #endregion

            using CSMI m = new CSMI();

            if (option_selected == 1)
            {
                for (int i = 1; i < 11; i++)
                {
                    try
                    {
                        Console.WriteLine("Iteration: " + i);
                        m.testRandom((int)Math.Pow(10, i));
                    }
                    catch (AcceleratorException e)
                    {
                        Console.WriteLine(e);
                        break;
                    }
                }
            }
            else
            {
                Console.WriteLine("Testing with reproducible data");
                m.testReproducible();
            }
        }
    }
}
