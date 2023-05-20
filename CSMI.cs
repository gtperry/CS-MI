using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Collections.Generic;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace CSMI
{
    public class Utils
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
    }

    public class MI : IDisposable
    {
        const double LOG_BASE = 2.0;
        const int MAX_YDIM = 65535;
        const int GRIDSIZE = 1024;
        Device dev;
        Context context;

        public MI()
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

        // Remember to instantiate MI class with a `using` statement. Called automatically.
        public void Dispose()
        {
            this.context.Dispose();
        }

        public double[] refactorArray(double[] arr)
        {
            IDictionary<int, double> newarr = new Dictionary<int, double>();
            double[] refactoredarr = new double[arr.GetLength(0)];
            double count = 0.0;
            int current;
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                current = (int)Math.Floor(arr[i]);
                if (!newarr.ContainsKey(current))
                {
                    newarr.Add(current, count);
                    count += 1.0;
                }
                refactoredarr[i] = newarr[current];
            }
            //print1d(refactoredarr);
            return refactoredarr;
        }

        public double[] refactorToMinimizeSize(double[] arr)
        {
            IDictionary<int, double> newarr = new Dictionary<int, double>();
            IDictionary<int, double> freqarr = new Dictionary<int, double>();
            double[] refactoredarr = new double[arr.GetLength(0)];
            double count = 0.0;
            int current;
            double temp;
            bool allnan = true;
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                current = (int)Math.Floor(arr[i]);
                if (!freqarr.ContainsKey(current))
                {
                    freqarr.Add(current, 1.0);
                }
                else
                {
                    temp = freqarr[current];
                    freqarr.Remove(current);
                    freqarr.Add(current, temp + 1.0);
                }
                //refactoredarr[i] = newarr[current];
            }
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                current = (int)Math.Floor(arr[i]);
                if (!newarr.ContainsKey(current))
                {
                    if (freqarr[current] > 1.0)
                    {
                        newarr.Add(current, count);
                        count += 1.0;
                        allnan = false;
                    }
                    else
                    {
                        newarr.Add(current, Double.NaN);
                    }
                }
                refactoredarr[i] = newarr[current];
            }
            if (Double.IsNaN(refactoredarr[0]))
            {
                refactoredarr[0] = count + 1;
            }
            //Console.WriteLine("Refactored arr");
            //print1d(arr);
            //print1d(refactoredarr);
            //Console.ReadLine();
            return refactoredarr;
        }

        public double calculateEntropy(double[] dataVector)
        {
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var MVBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            using var MVNormBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            //var FreqBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            MVBuffer.CopyFromCPU(dataVector);
            using var MaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var MinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjKernel);

            var CalcEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcEntropyKernel);

            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            InitMaxMinKern(MaxVal.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            GetMaxValKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            normalizeArrayKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MVNormBuffer.View, MinVal);
            using var FreqBuffer = accelerate.Allocate1D<double>(
                new Index1D(MaxVal.GetAsArray1D()[0] - MinVal.GetAsArray1D()[0] + 1)
            );
            //setBuffToValueKern(FreqBuffer.Extent.ToIntIndex(), FreqBuffer.View, 0.0);
            // if(dataVector.GetLength(0) > MAX_YDIM){
            //     int zdim = (int)Math.Floor((double)MVBuffer.Extent.ToIntIndex().X/GRIDSIZE) + 1;
            //     int ydim = GRIDSIZE;
            //     Console.WriteLine("YDIM AND ZDIM");
            //     Console.WriteLine(ydim);
            //     Console.WriteLine(zdim);
            //     LargeBuildFreqKern(new Index3D(MVBuffer.Extent.ToIntIndex().X,ydim, zdim), MVBuffer.View, FreqBuffer.View, GRIDSIZE, dataVector.GetLength(0));
            // }
            // else{
            //BuildFreqKern(new Index2D(MVBuffer.Extent.ToIntIndex().X,MVBuffer.Extent.ToIntIndex().X), MVBuffer.View, FreqBuffer.View);
            BuildFreqKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, FreqBuffer.View);
            //}

            //print1d(FreqBuffer.GetAsArray1D());

            CalcEntropyKern(
                FreqBuffer.Extent.ToIntIndex(),
                FreqBuffer.View,
                EntropyBuffer.View,
                dataVector.GetLength(0)
            );
            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public double mergeArrays(double[] firstVector, double[] secondVector, ref double[] outputVector)
        {
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            using var outputBuffer = accelerate.Allocate1D<double>(new Index1D(outputVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var StateCount = accelerate.Allocate1D<int>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);
            var mergeArraysKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                int,
                int
            >(mergeArraysKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);

            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];

            using var StateMap = accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1) * (secondnumstates + 1)));
            //outputVector = outputBuffer.GetAsArray1D();
            mergeArraysKern(
                StateCount.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                StateMap.View,
                outputBuffer.View,
                StateCount.View,
                firstnumstates,
                FirstNormBuffer.Extent.ToIntIndex().X
            );
            outputVector = outputBuffer.GetAsArray1D();
            // print1d(outputVector);
            double answer = StateCount.GetAsArray1D()[0];
            accelerate.Dispose();
            return answer;
        }

        public double calculateJointEntropy(double[] firstVector, double[] secondVector)
        {
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            using var TestBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcJointEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcJointEntropyKernel);

            var IndexedCalcJointEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcJointEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjustedKernel);
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            //Console.WriteLine(firstnumstates);
            //Console.WriteLine(secondnumstates);

            using var JointBuffer = accelerate.Allocate2DDenseX<double>(
                new Index2D(firstnumstates + 1, secondnumstates + 1)
            );

            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );

            //CalcJointEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, EntropyBuffer.View, firstVector.GetLength(0));
            IndexedCalcJointEntropyKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0)
            );

            // Console.WriteLine("joint");
            // Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine(TestBuffer.GetAsArray1D()[0]);

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public double calculateConditionalEntropy(double[] firstVector, double[] secondVector)
        {
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            double answer;
            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            using var TestBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(TestGetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcConditionalEntropyKernel);

            var IndexedCalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcConditionalEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            //print1d(SecondBuffer.GetAsArray1D());
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            //print1d(SecondMinVal.GetAsArray1D());

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            // Console.WriteLine(firstnumstates);
            // Console.WriteLine(secondnumstates);

            using var JointBuffer = accelerate.Allocate2DDenseX<double>(
                new Index2D(firstnumstates + 1, secondnumstates + 1)
            );
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            using var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            using var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            // Console.WriteLine("First and second norm:");
            //print1d(SecondMinVal.GetAsArray1D());
            //print1d(SecondNormBuffer.GetAsArray1D());
            //Console.ReadLine();

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );
            //print2d(JointBuffer.GetAsArray2D());
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine("JOINT");
            //print2d(JointBuffer.GetAsArray2D());
            // Console.WriteLine("SecondCountMap");
            //print1d(SecondCountMap.GetAsArray1D());
            // /CalcConditionalEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            IndexedCalcConditionalEntropyKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                SecondCountMap.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0)
            );

            // Console.WriteLine("TESTTT");
            // Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine(TestBuffer.GetAsArray1D()[0]);

            // Console.WriteLine("ENTROPY ARR");
            // print1d(EntropyBuffer.GetAsArray1D());
            //double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public double calculateConditionalEntropyAdjusted(double[] firstVector, double[] secondVector, int nanvals)
        {
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            double answer;
            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            using var TestBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(TestGetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcConditionalEntropyKernel);

            var IndexedCalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcConditionalEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjKernel);
            // Console.WriteLine("NAN FLOOR");
            // Console.WriteLine(Math.Floor(Double.NaN));
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            //print1d(SecondMinVal.GetAsArray1D());
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            //print1d(SecondMinVal.GetAsArray1D());

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            // Console.WriteLine(firstnumstates);
            // Console.WriteLine(secondnumstates);

            using var JointBuffer = accelerate.Allocate2DDenseX<double>(
                new Index2D(firstnumstates + 1, secondnumstates + 1)
            );
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            using var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            using var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            // Console.WriteLine("First and second norm:");
            // print1d(SecondMinVal.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            // Console.ReadLine();

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );
            //print2d(JointBuffer.GetAsArray2D());
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine("JOINT");
            //print2d(JointBuffer.GetAsArray2D());
            // Console.WriteLine("SecondCountMap");
            //print1d(SecondCountMap.GetAsArray1D());
            // /CalcConditionalEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            IndexedCalcConditionalEntropyKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                SecondCountMap.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0)
            );

            // Console.WriteLine("TESTTT");
            // Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine(TestBuffer.GetAsArray1D()[0]);

            // Console.WriteLine("ENTROPY ARR");
            // print1d(EntropyBuffer.GetAsArray1D());
            //double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public double calculateMutualInformation(double[] firstVector, double[] secondVector)
        {
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcMIKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcMIKernel);

            var IndexedCalcMIKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcMIKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjustedKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );

            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            //Console.WriteLine(firstnumstates);
            //Console.WriteLine(secondnumstates);

            using var JointBuffer = accelerate.Allocate2DDenseX<double>(
                new Index2D(firstnumstates + 1, secondnumstates + 1)
            );

            using var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            using var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            BuildFreqKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstCountMap.View);

            //CalcMIKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));

            IndexedCalcMIKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                FirstCountMap.View,
                SecondCountMap.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0)
            );

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public List<double> MulticalculateMutualInformation(List<double[]> arr)
        {
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));

            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(1, 1));
            var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(1));
            var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(1));
            int firstnumstates;
            int secondnumstates;
            double answer;

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);

            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcMIKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcMIKernel);

            var IndexedCalcMIKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcMIKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjustedKernel);
            List<double> MIanswers = new List<double>();

            for (int i = 0; i < arr.Count; i++)
            {
                for (int j = 0; j < arr.Count; j++)
                {
                    if (i != j)
                    {
                        FirstBuffer.CopyFromCPU(arr[i]);
                        SecondBuffer.CopyFromCPU(arr[j]);
                        InitMaxMinKern(
                            FirstMaxVal.Extent.ToIntIndex(),
                            FirstBuffer.View,
                            FirstMaxVal.View,
                            FirstMinVal.View
                        );
                        GetMaxValKern(
                            FirstBuffer.Extent.ToIntIndex(),
                            FirstBuffer.View,
                            FirstMaxVal.View,
                            FirstMinVal.View
                        );
                        //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
                        InitMaxMinKern(
                            SecondMaxVal.Extent.ToIntIndex(),
                            SecondBuffer.View,
                            SecondMaxVal.View,
                            SecondMinVal.View
                        );
                        GetMaxValKern(
                            SecondBuffer.Extent.ToIntIndex(),
                            SecondBuffer.View,
                            SecondMaxVal.View,
                            SecondMinVal.View
                        );

                        normalizeArrayKern(
                            FirstBuffer.Extent.ToIntIndex(),
                            FirstBuffer.View,
                            FirstNormBuffer.View,
                            FirstMinVal
                        );
                        normalizeArrayKern(
                            SecondBuffer.Extent.ToIntIndex(),
                            SecondBuffer.View,
                            SecondNormBuffer.View,
                            SecondMinVal
                        );

                        firstnumstates = FirstMaxVal.GetAsArray1D()[0];
                        secondnumstates = SecondMaxVal.GetAsArray1D()[0];
                        //Console.WriteLine(firstnumstates);
                        //Console.WriteLine(secondnumstates);

                        JointBuffer = accelerate.Allocate2DDenseX<double>(
                            new Index2D(firstnumstates + 1, secondnumstates + 1)
                        );

                        FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
                        SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
                        //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
                        // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
                        // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
                        setBuffToValueDoubleKern(FirstCountMap.Extent.ToIntIndex(), FirstCountMap.View, 0.0);
                        setBuffToValueDoubleKern(FirstCountMap.Extent.ToIntIndex(), FirstCountMap.View, 0.0);
                        BuildJointFreqKern(
                            SecondNormBuffer.Extent.ToIntIndex(),
                            FirstNormBuffer.View,
                            SecondNormBuffer.View,
                            JointBuffer.View
                        );
                        BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
                        BuildFreqKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstCountMap.View);

                        //CalcMIKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
                        setBuffToValueDoubleKern(EntropyBuffer.Extent.ToIntIndex(), EntropyBuffer.View, 0.0);
                        IndexedCalcMIKern(
                            SecondNormBuffer.Extent.ToIntIndex(),
                            JointBuffer.View,
                            FirstCountMap.View,
                            SecondCountMap.View,
                            FirstNormBuffer.View,
                            SecondNormBuffer.View,
                            EntropyBuffer.View,
                            arr[0].GetLength(0)
                        );

                        answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
                        MIanswers.Add(answer);
                    }
                }
            }
            JointBuffer.Dispose();
            FirstCountMap.Dispose();
            SecondCountMap.Dispose();
            accelerate.Dispose();
            return MIanswers;
        }

        public double calculateConditionalMutualInformation(
            double[] firstVector,
            double[] secondVector,
            double[] conditionVector
        )
        {
            //FirstCondEntropy (secondvector, conditionvector)
            //SecondCondEntropy (secondvector, mergedvector)
            Stopwatch watch = new Stopwatch();
            watch.Start();
            using Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            using var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var CondBuffer = accelerate.Allocate1D<double>(new Index1D(conditionVector.GetLength(0)));
            var testIndex = new LongIndex2D(10000, 1000000);
            using var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            using var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            using var CondNormBuffer = accelerate.Allocate1D<double>(new Index1D(conditionVector.GetLength(0)));

            using var mergedBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            //var AdjMergeBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));

            using var MergeNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);
            CondBuffer.CopyFromCPU(conditionVector);

            using var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var CondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var CondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var MergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var MergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var PreMergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            using var PreMergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            using var EntropyBuffer1 = accelerate.Allocate1D<double>(new Index1D(1));
            using var EntropyBuffer2 = accelerate.Allocate1D<double>(new Index1D(1));

            using var StateCount = accelerate.Allocate1D<int>(new Index1D(1));

            using var CountNonNaN = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(TestGetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(InitMaxMinKernel);
            var normalizeArrayKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>
            >(normalizeArrayKernel);
            var mergeArraysKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                int,
                int
            >(mergeArraysAdjKernel);
            var BuildJointFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>
            >(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double
            >(setBuffToValue2DKernal);

            var CalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(CalcConditionalEntropyKernel);

            var IndexedCalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(IndexedCalcConditionalEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double
            >(setBuffToValueDoubleKernal);

            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int
            >(setBuffToValueKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(BuildFreqAdjKernel);

            var RefactorPart1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(refactorKernal);

            var refactorPart1Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(refactorPart1Phase1Kernal);

            var refactorPart1Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(refactorPart1Phase2Kernal);

            var refactorPart2Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(refactorPart2Phase1Kernal);

            var refactorPart2Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(refactorPart2Phase2Kernal);

            var countnonNaNKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
            >(countnonNaN);

            // watch.Stop();
            // Console.WriteLine("Initialization: ");
            // Console.WriteLine(watch.ElapsedMilliseconds);
            // watch.Reset();
            // accelerate.DefaultStream.Synchronize();
            // watch.Start();
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);

            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            InitMaxMinKern(CondMaxVal.Extent.ToIntIndex(), CondBuffer.View, CondMaxVal.View, CondMinVal.View);
            GetMaxValKern(CondBuffer.Extent.ToIntIndex(), CondBuffer.View, CondMaxVal.View, CondMinVal.View);

            normalizeArrayKern(
                FirstBuffer.Extent.ToIntIndex(),
                FirstBuffer.View,
                FirstNormBuffer.View,
                FirstMinVal.View
            );
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal.View
            );
            normalizeArrayKern(CondBuffer.Extent.ToIntIndex(), CondBuffer.View, CondNormBuffer.View, CondMinVal.View);
            // accelerate.DefaultStream.Synchronize();

            // watch.Stop();
            // //RefactorPart1TestKern(testIndex, FirstBuffer.View, SecondBuffer.View);
            // Console.WriteLine("MaxMin ETC: ");

            // Console.WriteLine(watch.ElapsedMilliseconds);
            // watch.Reset();

            // accelerate.DefaultStream.Synchronize();
            // watch.Start();

            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            int condnumstates = CondMaxVal.GetAsArray1D()[0];

            using var StateMap = accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1) * (secondnumstates + 1)));
            //outputVector = mergedBuffer.GetAsArray1D();

            setBuffToValueKern(StateCount.Extent.ToIntIndex(), StateCount.View, 1);
            //print1d(FirstNormBuffer.GetAsArray1D());
            //print1d(SecondNormBuffer.GetAsArray1D());
            //Console.ReadLine();
            mergeArraysKern(
                mergedBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                StateMap.View,
                mergedBuffer.View,
                StateCount.View,
                firstnumstates,
                FirstNormBuffer.Extent.ToIntIndex().X
            );
            // mergeArraysKern(StateCount.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, StateMap.View, mergedBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);

            //outputVector = mergedBuffer.GetAsArray1D();
            // print1d(outputVector);
            // Console.WriteLine("InHere");
            // print1d(SecondNormBuffer.GetAsArray1D());
            // print1d(FirstNormBuffer.GetAsArray1D());
            //print1d(mergedBuffer.GetAsArray1D());
            // Console.ReadLine();
            InitMaxMinKern(
                PreMergeMaxVal.Extent.ToIntIndex(),
                mergedBuffer.View,
                PreMergeMaxVal.View,
                PreMergeMinVal.View
            );
            GetMaxValKern(
                mergedBuffer.Extent.ToIntIndex(),
                mergedBuffer.View,
                PreMergeMaxVal.View,
                PreMergeMinVal.View
            );
            //accelerate.DefaultStream.Synchronize();
            int premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];
            using var TempBuffer = accelerate.Allocate1D<double>(new Index1D(premergenumstates + 1));
            //var Temp2Buffer = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));

            using var HolderBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            //accelerate.DefaultStream.Synchronize();
            // print1d(PreMergeMinVal.GetAsArray1D());
            // Console.ReadLine();
            // Console.WriteLine("HERHEREHERHERHEHREREH");

            // watch.Stop();

            //Console.WriteLine("Merge Arrs:");
            //Console.WriteLine(watch.ElapsedMilliseconds);
            //accelerate.DefaultStream.Synchronize();

            //watch.Reset();
            //watch.Start();

            ///WILL PUT THIS IN THE GPU
            //RefactorPart1Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, firstVector.GetLength(0));
            refactorPart1Phase1Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);
            //Console.WriteLine(premergenumstates);
            //print1d(PreMergeMaxVal.GetAsArray1D());
            //Console.ReadLine();
            //print1d(TempBuffer.GetAsArray1D());

            refactorPart1Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);
            //accelerate.DefaultStream.Synchronize();
            //print1d(mergedBuffer.GetAsArray1D());
            watch.Stop();
            //print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();

            // Console.WriteLine("Part 1:");
            // Console.WriteLine(watch.ElapsedMilliseconds);
            // Console.ReadLine();

            watch.Reset();
            //print1d(mergedBuffer.GetAsArray1D());
            //Console.WriteLine("Testtiong");
            //Console.ReadLine();
            watch.Start();
            setBuffToValueDoubleKern(TempBuffer.Extent.ToIntIndex(), TempBuffer.View, 1.0);

            setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

            refactorPart2Phase1Kern(
                mergedBuffer.Extent.ToIntIndex(),
                mergedBuffer.View,
                TempBuffer.View,
                HolderBuffer.View
            );
            //print1d(TempBuffer.GetAsArray1D());
            //print1d(HolderBuffer.GetAsArray1D());
            //Console.ReadLine();

            refactorPart2Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);

            //print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();
            //AdjMergeBuffer.CopyFromCPU(refactorToMinimizeSize(mergedBuffer.GetAsArray1D()));

            //print1d(mergedBuffer.GetAsArray1D());
            //accelerate.DefaultStream.Synchronize();

            watch.Stop();

            // Console.WriteLine("Refactor Arrs Part2:");

            //Console.WriteLine(watch.ElapsedMilliseconds);
            watch.Reset();
            InitMaxMinKern(MergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            //print1d(MergeMaxVal.GetAsArray1D());

            // print1d(MergeMaxVal.GetAsArray1D());
            // print1d(MergeMinVal.GetAsArray1D());
            // print1d(mergedBuffer.GetAsArray1D());
            // Console.ReadLine();

            normalizeArrayKern(
                mergedBuffer.Extent.ToIntIndex(),
                mergedBuffer.View,
                MergeNormBuffer.View,
                MergeMinVal.View
            );

            int mergenumstates = MergeMaxVal.GetAsArray1D()[0];

            setBuffToValueDoubleKern(CountNonNaN.Extent.ToIntIndex(), CountNonNaN.View, 1.0);
            countnonNaNKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, CountNonNaN.View);

            // Console.WriteLine("NumStates");
            // Console.WriteLine(secondnumstates);

            // Console.WriteLine(condnumstates);

            // Console.WriteLine(mergenumstates);
            // Console.WriteLine(MergeMinVal.GetAsArray1D()[0]);
            // Console.WriteLine("PreMergeNumStates");
            // Console.WriteLine(premergenumstates);

            // Console.WriteLine(CountNonNaN.GetAsArray1D()[0]);

            //print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();

            using var JointBuffer1 = accelerate.Allocate2DDenseX<double>(
                new Index2D(secondnumstates + 1, condnumstates + 1)
            );
            using var JointBuffer2 = accelerate.Allocate2DDenseX<double>(
                new Index2D(secondnumstates + 1, mergenumstates + 1)
            );

            //EntropyBuffer1.GetAsArray1D();

            //var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            using var CondCountMap = accelerate.Allocate1D<double>(new Index1D(condnumstates));
            using var MergeCountMap = accelerate.Allocate1D<double>(new Index1D(mergenumstates));

            // print1d(CondNormBuffer.GetAsArray1D());
            // print1d(MergeNormBuffer.GetAsArray1D());

            BuildJointFreqKern(
                CondNormBuffer.Extent.ToIntIndex(),
                SecondNormBuffer.View,
                CondNormBuffer.View,
                JointBuffer1.View
            );
            //EntropyBuffer1.GetAsArray1D();
            //EntropyBuffer1.GetAsArray1D();

            BuildJointFreqKern(
                MergeNormBuffer.Extent.ToIntIndex(),
                SecondNormBuffer.View,
                MergeNormBuffer.View,
                JointBuffer2.View
            );
            //EntropyBuffer1.GetAsArray1D();

            //EntropyBuffer1.GetAsArray1D();

            BuildFreqKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, CondCountMap.View);
            //EntropyBuffer1.GetAsArray1D();
            //print1d(MergeNormBuffer.GetAsArray1D());
            BuildFreqKern(MergeNormBuffer.Extent.ToIntIndex(), MergeNormBuffer.View, MergeCountMap.View);
            //EntropyBuffer1.GetAsArray1D();

            //EntropyBuffer1.GetAsArray1D();
            IndexedCalcConditionalEntropyKern(
                CondNormBuffer.Extent.ToIntIndex(),
                JointBuffer1.View,
                CondCountMap.View,
                SecondNormBuffer.View,
                CondNormBuffer.View,
                EntropyBuffer1.View,
                secondVector.GetLength(0)
            );
            IndexedCalcConditionalEntropyKern(
                MergeNormBuffer.Extent.ToIntIndex(),
                JointBuffer2.View,
                MergeCountMap.View,
                SecondNormBuffer.View,
                MergeNormBuffer.View,
                EntropyBuffer2.View,
                secondVector.GetLength(0)
            );

            double ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            double ent2 = EntropyBuffer2.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            return ent1 - ent2;
        }

        public double ConditiponalMIHolder(double[] firstVector, double[] secondVector, double[] conditionVector)
        {
            double[] mergedVector = new double[firstVector.GetLength(0)];

            mergeArrays(firstVector, conditionVector, ref mergedVector);

            //Secondvector = first vector
            // conditionvector = second vector
            double firstCondEnt = calculateConditionalEntropy(secondVector, conditionVector);
            //print1d(refactorToMinimizeSize(mergedVector));

            //second vector = first vector
            // merged vector = second vector
            double secondCondEnt = calculateConditionalEntropy(secondVector, refactorToMinimizeSize(mergedVector));

            return firstCondEnt - secondCondEnt;
        }

        static void replaceNaNKernal(Index1D index, ArrayView1D<double, Stride1D.Dense> inputView)
        {
            if (Double.IsNaN(inputView[index]))
            {
                inputView[index] = -1.0;
            }
        }

        static void refactorKernal(Index1D index, ArrayView1D<double, Stride1D.Dense> inputView, int length)
        {
            int holder = 0;

            for (int i = 0; i < length; i++)
            {
                if (inputView[index] == inputView[new Index1D(i)] && index.X != i)
                {
                    holder = 1;
                }
            }
            if (holder == 0)
            {
                inputView[index] = Double.NaN;
            }
        }

        // static void fixFirst(Index1D index,
        //     ArrayView1D<double, Stride1D.Dense> inputView){
        //     if(Double.IsNaN(inputView[index]))
        // }
        static void refactorPart1Phase1Kernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView
        )
        {
            if (!Double.IsNaN(inputView[index]) && inputView[index] > 1000)
            {
                Atomic.Add(ref holderView[new Index1D((int)inputView[index])], 1.0);
            }
        }

        static void refactorPart1Phase2Kernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView
        )
        {
            if (holderView[new Index1D((int)inputView[index])] <= 1.0)
            {
                inputView[index] = Double.NaN;
            }
        }

        static void refactorPart2Phase1Kernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView,
            ArrayView1D<double, Stride1D.Dense> sharedmem
        )
        {
            if (!Double.IsNaN(inputView[index]))
            {
                // //double x = Atomic.Add(ref holderView[(int)inputView[index]], Atomic.Add(ref sharedmem[new Index1D(0)], 1.0));
                // if(holderView[(int)inputView[index]] > 1.0){
                //     //holderView[(int)inputView[index]] = x;
                //     //Atomic.Add(ref holderView[(int)inputView[index]], sharedmem[new Index1D(0)]);
                // }
                // else{
                holderView[(int)inputView[index]] = Atomic.Add(ref sharedmem[new Index1D(0)], 1.0);
            }
        }

        static void countnonNaN(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            if (!Double.IsNaN(inputView[index]))
            {
                Atomic.Add(ref output[new Index1D(0)], 1.0);
            }
        }

        static void refactorPart2Phase2Kernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView
        )
        {
            if (!Double.IsNaN(inputView[index]))
            {
                inputView[index] = holderView[(int)inputView[index]];
            }
            else if (index.X == 0 && Double.IsNaN(inputView[index]))
            {
                inputView[new Index1D(0)] = 1.0;
            }
        }

        // static void refactorPart2Kernal(Index1D index,
        //     ArrayView1D<double, Stride1D.Dense> inputView,
        //     ArrayView1D<double, Stride1D.Dense> holderView,
        //     ArrayView1D<double, Stride1D.Dense> sharedmem){

        //     if(!Double.IsNaN(inputView[index])){
        //         inputView[index] = Atomic.Add(ref sharedmem[new Index1D(0)], 1.0);

        //     }
        //     else if(index.X == 0 && Double.IsNaN(inputView[index] )){
        //         inputView[new Index1D(0)] = Atomic.Add(ref sharedmem[new Index1D(0)], 1.0);
        //     }
        // }

        // static void refactorPart2Kernale(Index1D index,
        //     ArrayView1D<double, Stride1D.Dense> inputView,
        //     int length){
        //     double temp = 1.0;
        //     bool needfront = true;
        //     for(int i = 0; i < length; i ++){

        //         if(!Double.IsNaN(inputView[new Index1D(i)])){
        //             inputView[new Index1D(i)] = temp;
        //             temp += 1.0;

        //         }

        //     }
        //     if(Double.IsNaN(inputView[new Index1D(0)])){
        //         inputView[new Index1D(0)] = temp;
        //     }
        // }

        static void mergeArraysKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> FirstNormView,
            ArrayView1D<double, Stride1D.Dense> SecondNormView,
            ArrayView1D<int, Stride1D.Dense> StateMap,
            ArrayView1D<double, Stride1D.Dense> OutputView,
            ArrayView1D<int, Stride1D.Dense> SCount,
            int firstnumstates,
            int length
        )
        {
            int curindex;
            int statecount = 1;
            for (int i = 0; i < length; i++)
            {
                curindex = (int)FirstNormView[new Index1D(i)] + ((int)SecondNormView[new Index1D(i)] * firstnumstates);
                if (StateMap[new Index1D(curindex)] == 0)
                {
                    StateMap[new Index1D(curindex)] = statecount;
                    statecount += 1;
                }
                OutputView[new Index1D(i)] = (double)StateMap[curindex];
            }
            SCount[index] = statecount;
        }

        static void mergeArraysAdjKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> FirstNormView,
            ArrayView1D<double, Stride1D.Dense> SecondNormView,
            ArrayView1D<int, Stride1D.Dense> StateMap,
            ArrayView1D<double, Stride1D.Dense> OutputView,
            ArrayView1D<int, Stride1D.Dense> SCount,
            int firstnumstates,
            int length
        )
        {
            int curindex;
            //int statecount = 1;
            //Probably doesnt work
            curindex = (int)FirstNormView[index] + ((int)SecondNormView[index] * firstnumstates);
            // if(Atomic.Add(ref StateMap[new Index1D(curindex)], 1) == 0){
            //     StateMap[new Index1D(curindex)] = Atomic.Add(ref SCount[new Index1D(0)], 1);
            // }
            // else{
            //     Atomic.Add(ref StateMap[new Index1D(curindex)], -1);
            // }
            if (StateMap[new Index1D(curindex)] == 0)
            {
                StateMap[new Index1D(curindex)] = Atomic.Add(ref SCount[new Index1D(0)], 1);
            }
            OutputView[index] = (double)StateMap[curindex];
        }

        static void CalcEntropyKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> FreqView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            if (FreqView[index] > 0)
            {
                val = -1 * (FreqView[index] / (double)length) * Math.Log(FreqView[index] / (double)length);
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void CalcConditionalEntropyKernel(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> condView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            if (FreqView[index] > 0 && condView[index.Y] > 0)
            {
                val =
                    -1
                    * (
                        (FreqView[index] / (double)length)
                        * Math.Log((FreqView[index] / (double)length) / (condView[index.Y] / (double)length))
                    );
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void IndexedCalcConditionalEntropyKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> condView,
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            if (!(Double.IsNaN(FirstView[index])) && !(Double.IsNaN(SecondView[index])))
            {
                Index2D ind2 = new Index2D((int)FirstView[index], (int)SecondView[index]);

                if (FreqView[ind2] > 0 && condView[ind2.Y] > 0)
                {
                    val =
                        -1
                        * (
                            (FreqView[ind2] / (double)length)
                            * Math.Log((FreqView[ind2] / (double)length) / (condView[ind2.Y] / (double)length))
                        );
                    val = val / FreqView[ind2];
                    Atomic.Add(ref entropy[new Index1D(0)], val);
                }
            }
            else
            {
                val = -1 * ((1 / (double)length) * Math.Log((1 / (double)length) / (1 / (double)length)));

                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void CalcMIKernel(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            if (FreqView[index] > 0 && first[index.X] > 0 && second[index.Y] > 0)
            {
                val = (
                    (FreqView[index] / (double)length)
                    * Math.Log(
                        ((FreqView[index] / (double)length) / (first[index.X] / (double)length))
                            / (second[index.Y] / (double)length)
                    )
                );
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void IndexedCalcMIKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            Index2D ind2 = new Index2D((int)FirstView[index], (int)SecondView[index]);
            if (FreqView[ind2] > 0 && first[(int)FirstView[index]] > 0 && second[(int)SecondView[index]] > 0)
            {
                // csharpier-ignore
                val = ((FreqView[ind2] / (double)length) * Math.Log(((FreqView[ind2] / (double)length) / (first[ind2.X] / (double)length)) / (second[ind2.Y] / (double)length)));
                val = val / FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void CalcJointEntropyKernel(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            if (FreqView[index] > 0)
            {
                val = -1 * (FreqView[index] / (double)length) * Math.Log((FreqView[index] / (double)length));
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void IndexedCalcJointEntropyKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> FreqView,
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length
        )
        {
            double val;
            Index2D ind2 = new Index2D((int)FirstView[index], (int)SecondView[index]);
            if (FreqView[ind2] > 0)
            {
                val = -1 * (FreqView[ind2] / (double)length) * Math.Log((FreqView[ind2] / (double)length));
                val = val / FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void normalizeArrayKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> outputView,
            ArrayView1D<int, Stride1D.Dense> minVal
        )
        {
            outputView[index] = Math.Floor(inputView[index]) - (double)minVal[new Index1D(0)];
        }

        static void GetMaxMinValKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal
        )
        {
            if (aView[index] != Double.NaN)
            {
                Atomic.Max(ref MaxVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                Atomic.Min(ref MinVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                //Atomic.Add(ref MinVal[new Index1D(0)], 1);
            }
        }

        static void TestGetMaxMinValKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal
        )
        {
            if (!Double.IsNaN(aView[index]))
            {
                Atomic.Max(ref MaxVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                Atomic.Min(ref MinVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                //Atomic.Add(ref MinVal[new Index1D(0)], 1);
            }
        }

        static void InsertBufferIntoTree(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> View,
            ArrayView1D<int, Stride1D.Dense> Tree
        ) { }

        static void InitMaxMinKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal
        )
        {
            MaxVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
            MinVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
        }

        ///<summary>Sets every element in buff to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        static void setBuffToValueKernal(Index1D index, ArrayView1D<int, Stride1D.Dense> buff, int setvalue)
        {
            buff[index] = setvalue;
        }

        ///<summary>Sets every element in buff to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        static void setBuffToValueDoubleKernal(Index1D index, ArrayView1D<double, Stride1D.Dense> buff, double setvalue)
        {
            buff[index] = setvalue;
        }

        ///<summary>Sets every element in buff to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        static void setBuffToValue2DKernal(Index2D index, ArrayView2D<double, Stride2D.DenseX> buff, double setvalue)
        {
            buff[index] = setvalue;
        }

        static void BuildFreqKernel(
            Index2D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            if (Math.Floor(input[index.X]) == Math.Floor(input[index.Y]))
            {
                Atomic.Add(ref output[index.X], 1.0);
            }
        }

        static void BuildFreqAdjKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            if (!(Double.IsNaN(input[index])))
            {
                Atomic.Add(ref output[(int)Math.Floor(input[index])], 1.0);
            }
            //if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y])){
            //}
        }

        static void LargeBuildFreqKernel(
            Index3D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output,
            int gridsize,
            int length
        )
        {
            if ((index.Z * gridsize) + index.Y < length)
            {
                if (Math.Floor(input[index.X]) == Math.Floor(input[(index.Z * gridsize) + index.Y]))
                {
                    Atomic.Add(ref output[index.X], 1.0);
                }
            }
        }

        static void BuildFreqAdjustedKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            //if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y]) && Math.Floor(input[index.X]) != 0){
            Atomic.Add(ref output[new Index1D((int)(input[index]))], 1.0);
            //}
        }

        static void BuildJointFreqKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView2D<double, Stride2D.DenseX> output
        )
        {
            if (!(Double.IsNaN(first[index])) && !(Double.IsNaN(second[index])))
            {
                //if(Math.Floor(first[index.X]) == Math.Floor(first[index.Y]) && Math.Floor(second[index.X]) == Math.Floor(second[index.Y])){
                Atomic.Add(ref output[new Index2D((int)first[index], (int)second[index])], 1.0);
            }
            //}
            // if(Math.Floor(first[index.X]) == Math.Floor(first[index.Y])){
            //     Atomic.Add(ref firstmap[index.X], 1.0);
            // }

            // if(Math.Floor(second[index.X]) == Math.Floor(second[index.Y])){
            //     Atomic.Add(ref secondmap[index.X], 1.0);
            // }
        }

        void print1d(int[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");
        }

        void print1d(double[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");
        }

        void print2d(double[,] array)
        {
            Console.WriteLine(array);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("{0}, ", array[i, j]);
                }
                Console.Write("]");
                Console.WriteLine(", ");
            }
            Console.WriteLine("]");
        }

        // void testMulti( List<double[]> arr){
        //     List<double> jointEntropyList = new List<double>();
        //     List<double> ConditionalMIList = new List<double>();

        //     Stopwatch timer = new Stopwatch();
        //     for(int i = 0; i < arr.GetLength(1); i++){
        //         for(int j = 0; j < arr.GetLength(1); j++){
        //             if(i != j){
        //                 jointEntropyList.Add(calculateJointEntropy(arr[i],arr[j]));
        //                 for(int k = 0; k < arr.GetLength(1); k++){
        //                     if (k != j && k != i){
        //                         ConditionalMIList.Add(calculateConditionalMutualInformation(arr[i],arr[j], arr[k]));
        //                     }
        //                 }
        //             }

        //         }
        //     }
        //     return (jointEntropyList, ConditionalMIList);
        // }

        double[] GenerateRandomNumbers(int length)
        {
            Random rand = new Random();
            double[] numbers = new double[length];

            for (int i = 0; i < length; i++)
            {
                numbers[i] = rand.NextDouble() * 10000;
            }

            return numbers;
        }

        /// <summary>
        /// For reference, the values returned for the functions here  (at double precision) in Java are the following:
        /// <para>calculateEntropy: 2.4464393446710155</para>
        /// <para>calculateConditionalEntropy: 0.6</para>
        /// <para>calculateJointEntropy: 3.121928094887362</para>
        /// <para>calculateMutualInformation1.8464393446710157</para>
        /// <para>calculateConditionalMutualInformation: 0.7509775004326935</para>
        /// </summary>
        void testReproducible()
        {
            // Reproducible data
            double[] a = new[] { 4.2, 5.43, 3.221, 7.34235, 1.931, 1.2, 5.43, 8.0, 7.34235, 1.931 };
            double[] b = new[] { 2.2, 3.43, 1.221, 9.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 34.931 };
            double[] c = new[] { 2.2, 3.43, 2.221, 2.34235, 3.931, 3.2, 4.43, 7.0, 7.34235, 34.931 };
            Console.WriteLine("Testing with reproducible data");
            // csharpier-ignore-start
            Utils.MeasureExecutionTime("Calculate Entropy", () => calculateEntropy(a), printOutput: true);
            Utils.MeasureExecutionTime("Calculate Conditional Entropy", () => calculateConditionalEntropy(a, b), printOutput: true);
            Utils.MeasureExecutionTime("Calculate Joint Entropy", () => calculateJointEntropy(a, b), printOutput: true);
            Utils.MeasureExecutionTime("Mutual Information", () => calculateMutualInformation(a, b), printOutput: true);
            Utils.MeasureExecutionTime("Conditional Mutual Information", () => calculateConditionalMutualInformation(a, b, c), printOutput: true);
            // csharpier-ignore-end

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

            Utils.MeasureExecutionTime("Mutual Information", () => calculateMutualInformation(a, b));
            Utils.MeasureExecutionTime("Calculate Entropy", () => calculateEntropy(a));
            Utils.MeasureExecutionTime("Calculate Conditional Entropy", () => calculateConditionalEntropy(a, b));
            Utils.MeasureExecutionTime("Calculate Joint Entropy", () => calculateJointEntropy(a, b));
            Utils.MeasureExecutionTime(
                "Conditional Mutual Information",
                () => calculateConditionalMutualInformation(a, b, c)
            );

            Console.WriteLine();
            Console.WriteLine("----------------------------");
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            using MI m = new MI();

            m.testReproducible();

            // for (int i = 1; i < 11; i++)
            // {
            //     try
            //     {
            //         Console.WriteLine("Iteration: " + i);
            //         m.testRandom((int)Math.Pow(10, i));
            //     }
            //     catch (AcceleratorException e)
            //     {
            //         Console.WriteLine(e);
            //         break;
            //     }
            // }
        }
    }
}
