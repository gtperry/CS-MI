using System;

using ILGPU;
using ILGPU.Runtime;

// using CK = CSMI.CommonKernels;

namespace CSMI
{
    public class Entropy : CommonKernels
    {
        private ILGPUInitializer gpu;

        public Entropy(ILGPUInitializer ilgpu)
        {
            gpu = ilgpu;
        }

        #region Methods
        public double calculateEntropy(double[] dataVector)
        {
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
            // BuildFreqKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, FreqBuffer.View);
            //}

            BuildFreqKern(MVNormBuffer.Extent.ToIntIndex(), MVNormBuffer.View, FreqBuffer.View);

            CalcEntropyKern(
                FreqBuffer.Extent.ToIntIndex(),
                FreqBuffer.View,
                EntropyBuffer.View,
                dataVector.GetLength(0)
            );
            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            // print1d(MVNormBuffer.GetAsArray1D());
            accelerate.Dispose();
            return answer;
        }

        public double calculateJointEntropy(double[] firstVector, double[] secondVector)
        {
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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

            // csharpier-ignore-start
            // Reusing these variables from before because we don't need the original values anymore now that everything is normalized
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            // csharpier-ignore-end

            int firstMaxVal = FirstMaxVal.GetAsArray1D()[0];
            int secondMaxVal = SecondMaxVal.GetAsArray1D()[0];

            using var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstMaxVal + 1, secondMaxVal + 1));
            // Console.WriteLine($"JointBuffer Size: {JointBuffer.Length}");

            // setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            // print2d(JointBuffer.GetAsArray2D());

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

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        /// <summary>
        /// Alternate implementation of JointEntropy using a 1D Buffer (szudzik for hashing) instead of a 2D Buffer.
        /// </summary>
        /// <param name="firstVector"></param>
        /// <param name="secondVector"></param>
        /// <returns></returns>
        public double calculateJointEntropy2(double[] firstVector, double[] secondVector)
        {
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                long
            >(BuildJointFreqKernel2);

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
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                int,
                long
            >(IndexedCalcJointEntropyKernel2);

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
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(
                SecondBuffer.Extent.ToIntIndex(),
                SecondBuffer.View,
                SecondNormBuffer.View,
                SecondMinVal
            );
            // Console.WriteLine($"FirstNormBuffer: [{string.Join(", ", FirstNormBuffer.GetAsArray1D())}]");
            // Console.WriteLine($"SecondNormBuffer: [{string.Join(", ", SecondNormBuffer.GetAsArray1D())}]");

            // csharpier-ignore-start
            // Reusing these variables from before because we don't need the original values anymore now that everything is normalized
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            // csharpier-ignore-end

            int firstMaxVal = FirstMaxVal.GetAsArray1D()[0];
            int secondMaxVal = SecondMaxVal.GetAsArray1D()[0];
            // Console.WriteLine($"FirstMaxVal = {firstMaxVal:n0}");
            // Console.WriteLine($"SecondMaxVal = {secondMaxVal:n0}");
            int firstMinVal = FirstMinVal.GetAsArray1D()[0];
            int secondMinVal = SecondMinVal.GetAsArray1D()[0];
            // Console.WriteLine($"FirstMinVal = {firstMinVal:n0}");
            // Console.WriteLine($"SecondMinVal = {secondMinVal:n0}");

            long maxIndexVal = Utils.szudzikPair(firstMaxVal, secondMaxVal);
            byte maxOrderMagnitude = (byte)maxIndexVal.ToString().Length;
            // Console.WriteLine($"MaxPossiblePair = {maxIndexVal:n0} with {maxOrderMagnitude} digits");
            long minIndexVal = Utils.szudzikPair(firstMinVal, secondMinVal);
            // Console.WriteLine($"MinPossiblePair = {minIndexVal:n0}");
            using var JointBuffer = accelerate.Allocate1D<double>(maxIndexVal - minIndexVal);
            // JointBuffer.MemSetToZero();
            // Console.WriteLine($"JointBuffer Size: {JointBuffer.Length:n0}");

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View,
                minIndexVal
            );

            IndexedCalcJointEntropyKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0),
                minIndexVal
            );

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        // TODO: Function not finished, just testing for now
        public double calculateJointEntropy3(double[] firstVector, double[] secondVector)
        {
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
            >(BuildJointFreqKernel3);

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
            >(IndexedCalcJointEntropyKernel3);

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
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            // csharpier-ignore-start
            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            // csharpier-ignore-end

            int firstMaxVal = FirstMaxVal.GetAsArray1D()[0];
            int secondMaxVal = SecondMaxVal.GetAsArray1D()[0];
            Console.WriteLine($"FirstMaxVal = {firstMaxVal:n0}");
            Console.WriteLine($"SecondMaxVal = {secondMaxVal:n0}");

            long maxIndexVal = Utils.szudzikPair(firstMaxVal, secondMaxVal);
            byte maxOrderMagnitude = (byte)maxIndexVal.ToString().Length;
            long maxIndexValSqrt = (long)Math.Ceiling(Math.Sqrt(maxIndexVal));
            Console.WriteLine(
                $"MaxPossiblePair = {maxIndexVal:n0} with {maxOrderMagnitude} digits, with matrix rank {maxIndexValSqrt:n0}"
            );
            using var JointBuffer = accelerate.Allocate2DDenseX<double>(
                new LongIndex2D(maxIndexValSqrt, maxIndexValSqrt)
            );

            BuildJointFreqKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                JointBuffer.View
            );

            IndexedCalcJointEntropyKern(
                SecondNormBuffer.Extent.ToIntIndex(),
                JointBuffer.View,
                FirstNormBuffer.View,
                SecondNormBuffer.View,
                EntropyBuffer.View,
                firstVector.GetLength(0)
            );

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public double calculateConditionalEntropy(double[] firstVector, double[] secondVector)
        {
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
            // csharpier-ignore-start
            // Reusing these variables from before because we don't need the original values anymore now that everything is normalized
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            // csharpier-ignore-end

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
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);

            using var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            using var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
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
            // print2d(JointBuffer.GetAsArray2D());
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine("JOINT");
            //print2d(JointBuffer.GetAsArray2D());
            // Console.WriteLine("SecondCountMap");
            //print1d(SecondCountMap.GetAsArray1D());
            // /CalcConditionalEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
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
            //double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            accelerate.Dispose();
            return answer;
        }

        public double calculateConditionalEntropyAdjusted(double[] firstVector, double[] secondVector, int nanvals)
        {
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);

            using var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            using var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
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
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine("JOINT");
            //print2d(JointBuffer.GetAsArray2D());
            // Console.WriteLine("SecondCountMap");
            //print1d(SecondCountMap.GetAsArray1D());
            // /CalcConditionalEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
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
            //double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            accelerate.Dispose();
            return answer;
        }
        #endregion

        #region: Kernels
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

        public static void CalcConditionalEntropyKernel(
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

        public static void IndexedCalcConditionalEntropyKernel(
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

        static void IndexedCalcJointEntropyKernel2(
            Index1D index,
            ArrayView<double> FreqView,
            ArrayView<double> FirstView,
            ArrayView<double> SecondView,
            ArrayView<double> entropy,
            int length,
            long minSzudzikPair
        )
        {
            double val;
            var ind2 = Utils.szudzikPair(
                (int)FirstView[index] - minSzudzikPair,
                (int)SecondView[index] - minSzudzikPair
            );
            if (FreqView[ind2] > 0)
            {
                val = -1 * (FreqView[ind2] / (double)length) * Math.Log((FreqView[ind2] / (double)length));
                val = val / FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }

        static void IndexedCalcJointEntropyKernel3(
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

        #endregion
    }
}
