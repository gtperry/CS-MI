using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace CSMI
{
    /// <summary>
    /// Remember to instantiate this class with a `using` statement to ensure resources are disposed.
    /// </summary>
    public class TestSuite : IDisposable
    {
        public ILGPUWrapper ILGPUWrapper { get; private set; }
        public Entropy entropy { get; private set; }
        public MutualInformation mi { get; private set; }
        public List<string> failedTests { get; private set; } = new List<string>();

        public TestSuite()
        {
            ILGPUWrapper = new ILGPUWrapper();
            entropy = new Entropy(ILGPUWrapper);
            mi = new MutualInformation(ILGPUWrapper);
            // Quick calculation to get things started/initialized and measured time is not affected for the actual results
            // entropy.calculateEntropy(new double[] { 1, 2, 3 });
        }

        // Called automatically if using a `using` statement
        public void Dispose()
        {
            ILGPUWrapper.Dispose();
        }

        public void runAllTests()
        {
            var methods = GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(item => item.Name.StartsWith("t_") && !item.Name.ToLower().Contains("random"));

            foreach (var method in methods)
            {
                try
                {
                    Console.WriteLine();
                    Console.WriteLine($"Test: {method.Name}");
                    Console.WriteLine(new string('-', 15));
                    method.Invoke(this, null);
                }
                catch (ApplicationException)
                {
                    Console.WriteLine($"Can't run method {method.Name}");
                }
            }

            Console.WriteLine();
            Console.WriteLine($"Total failed tests: {failedTests.Count}");
            foreach (var failedTest in failedTests)
            {
                Console.WriteLine($"\t{failedTest}");
            }
        }

        public void testRandom(int length)
        {
            Console.WriteLine($"LENGTH = {length:n0}");
            double[] a = Utils.GenerateRandomNumbers(length);
            double[] b = Utils.GenerateRandomNumbers(length);
            double[] c = Utils.GenerateRandomNumbers(length);
            // csharpier-ignore-start
            Utils.MeasureExecutionTime("Calculate Entropy", () => entropy.calculateEntropy(a));
            Utils.MeasureExecutionTime("Calculate Conditional Entropy", () => entropy.calculateConditionalEntropy(a, b));
            Utils.MeasureExecutionTime("Calculate Joint Entropy", () => entropy.calculateJointEntropy(a, b));
            // Utils.MeasureExecutionTime("Calculate Joint Entropy 2", () => entropy.calculateJointEntropy2(a, b));
            // Utils.MeasureExecutionTime("Calculate Joint Entropy 3", () => entropy.calculateJointEntropy3(a, b)); // TODO: finish or remove
            Utils.MeasureExecutionTime("Mutual Information", () => mi.calculateMutualInformation(a, b));
            Utils.MeasureExecutionTime("Conditional Mutual Information", () => mi.calculateConditionalMutualInformation(a, b, c));
            // csharpier-ignore-end

            Console.WriteLine(Environment.NewLine + new string('-', 15) + Environment.NewLine);
        }

        /// <summary>
        /// This function runs tests against the Java implementation of this library to ensure that the results are reproducible.
        /// </summary>
        public void t_CSMIPositiveNumbers1()
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
        }

        protected void compareResults(
            double[] resultsList,
            Dictionary<int, (string name, double javaResult)> javaResultsMap,
            int decimalPlaces = 12,
            [CallerMemberName] string callerName = ""
        )
        {
            // See if the functions for the C# vs Java implementation are the same
            Console.WriteLine(
                $"{Environment.NewLine}Java vs C# return values test results, accurate to '{decimalPlaces}' decimal places:"
            );
            for (int i = 0; i < resultsList.Length; i++)
            {
                var csharpResult = Math.Round(resultsList[i], decimalPlaces);
                var javaResult = Math.Round(javaResultsMap[i].javaResult, decimalPlaces);
                string testOutcome;
                if ((csharpResult == javaResult))
                {
                    testOutcome = "PASS";
                }
                else
                {
                    testOutcome = "\x1b[41mFAIL\x1b[0m";
                    failedTests.Add($"{callerName}, {javaResultsMap[i].name} -> Expected: {javaResult}, Actual: {csharpResult}");
                }

                Console.WriteLine($"{i}) {javaResultsMap[i].name}: {testOutcome}");
            }
        }

        public void t_CSMIPositiveNumbers2()
        {
            double[] a = new[] { 1.2, 2.4, 1.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1 };
            double[] b = new[] { 2.2, 2.4, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1 };
            double[] c = new[] { 2.2, 3.43, 2.221, 2.34235, 3.931, 3.2, 4.43, 7.0, 7.34235, 34.931 };

            var javaResultsMap = new Dictionary<int, (string name, double javaResult)>()
            {
                { 0, ("Entropy", 3.121928094887362) },
                { 1, ("ConditionalEntropy", 0.27548875021634683) },
                { 2, ("JointEntropy", 3.121928094887362) },
                { 3, ("MutualInformation", 2.846439344671016) },
                { 4, ("ConditionalMutualInformation", 0.9509775004326936) },
            };
            // csharpier-ignore-start
            var resultsList = new double[]{
                Utils.MeasureExecutionTime(javaResultsMap[0].name, () => entropy.calculateEntropy(a), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[1].name, () => entropy.calculateConditionalEntropy(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[2].name, () => entropy.calculateJointEntropy(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[3].name, () => mi.calculateMutualInformation(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[4].name, () => mi.calculateConditionalMutualInformation(a, b, c), printOutput: true),
            };
            // csharpier-ignore-end

            compareResults(resultsList, javaResultsMap);
        }

        public void t_CSMINegativeNumbers()
        {
            // Fixed data for reproducible results
            double[] a = new[] { 4.2, -5.43, -3.221, -7.34235, -1.931, -1.2, -5.43, -8.0, -7.34235, -1.931 };
            double[] b = new[] { 2.2, 3.43, 1.221, 9.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 34.931 };
            double[] c = new[] { -2.2, 3.43, 2.221, 2.34235, 3.931, 3.2, 4.43, 7.0, 7.34235, -34.931 };

            // These are the results obtained from the Java implementation of this library for each function.
            var javaResultsMap = new Dictionary<int, (string name, double javaResult)>()
            {
                { 0, ("Entropy", 2.1709505944546685) },
                { 1, ("ConditionalEntropy", 0.39999999999999997) },
                { 2, ("JointEntropy", 2.9219280948873623) },
                { 3, ("MutualInformation", 1.7709505944546688) },
                { 4, ("ConditionalMutualInformation", 0.4754887502163468) },
            };
            // csharpier-ignore-start
            var resultsList = new double[]{
                Utils.MeasureExecutionTime(javaResultsMap[0].name, () => entropy.calculateEntropy(a), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[1].name, () => entropy.calculateConditionalEntropy(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[2].name, () => entropy.calculateJointEntropy(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[3].name, () => mi.calculateMutualInformation(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[4].name, () => mi.calculateConditionalMutualInformation(a, b, c), printOutput: true),
            };
            // csharpier-ignore-end

            compareResults(resultsList, javaResultsMap);
        }

        public void t_CSMIPositiveNumbersBigMagnitude()
        {
            double[] a = new[] { 4.2, 5.43, 3.221, 7.34235, 1.931, 1.2, 5.43, 8.0, 7.34235, 1.931 };
            double[] b = new[] { 2.2, 3.43, 1.221, 9.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 34.931 };
            double[] c = new[] { 2.2, 3.43, 2.221, 2.34235, 3.931, 3.2, 4.43, 7.0, 7.34235, 34.931 };

            // Make the values bigger
            int orderOfMagnitude = 5;
            int numToMultiply = (int)Math.Pow(10, orderOfMagnitude);
            for (int i = 0; i < a.Length; i++)
            {
                // Addition doesn't change the results, but multiplication does.
                a[i] += numToMultiply;
                b[i] += numToMultiply;
                c[i] += numToMultiply;
            }
            // Console.WriteLine($"a: [{string.Join(", ", a)}]");

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
                Utils.MeasureExecutionTime(javaResultsMap[3].name, () => mi.calculateMutualInformation(a, b), printOutput: true),
                Utils.MeasureExecutionTime(javaResultsMap[4].name, () => mi.calculateConditionalMutualInformation(a, b, c), printOutput: true),
            };
            // csharpier-ignore-end

            compareResults(resultsList, javaResultsMap);
        }
    }
}
