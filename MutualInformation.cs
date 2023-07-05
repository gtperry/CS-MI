using System;
using System.Collections.Generic;

using ILGPU;
using ILGPU.Runtime;

namespace CSMI
{
    public class MutualInformation : CommonKernels
    {
        private ILGPUWrapper gpu;

        public MutualInformation(ILGPUWrapper ilgpu)
        {
            this.gpu = ilgpu;
        }

        #region Methods
        public double calculateMutualInformation(double[] firstVector, double[] secondVector)
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
            // csharpier-ignore-start
            // Reusing these variables from before because we don't need the original values anymore now that everything is normalized
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            // csharpier-ignore-end

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

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
            accelerate.Dispose();
            return answer;
        }

        public List<double> MulticalculateMutualInformation(List<double[]> arr)
        {
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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

                        answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);
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
            using Accelerator accelerate = gpu.dev.CreateAccelerator(gpu.context);
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
            >(Entropy.CalcConditionalEntropyKernel);

            var IndexedCalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
            >(Entropy.IndexedCalcConditionalEntropyKernel);

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

            // csharpier-ignore-start
            // Reusing these variables from before because we don't need the original values anymore now that everything is normalized
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            InitMaxMinKern(CondMaxVal.Extent.ToIntIndex(), CondNormBuffer.View, CondMaxVal.View, CondMinVal.View);
            GetMaxValKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, CondMaxVal.View, CondMinVal.View);
            // csharpier-ignore-end

            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            int condnumstates = CondMaxVal.GetAsArray1D()[0];

            using var StateMap = accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1) * (secondnumstates + 1)));
            //outputVector = mergedBuffer.GetAsArray1D();

            setBuffToValueKern(StateCount.Extent.ToIntIndex(), StateCount.View, 1);
            //Utils.print1d(FirstNormBuffer.GetAsArray1D());
            //Utils.print1d(SecondNormBuffer.GetAsArray1D());
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
            // Utils.print1d(outputVector);
            // Console.WriteLine("InHere");
            // Utils.print1d(SecondNormBuffer.GetAsArray1D());
            // Utils.print1d(FirstNormBuffer.GetAsArray1D());
            //Utils.print1d(mergedBuffer.GetAsArray1D());
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
            // Utils.print1d(PreMergeMinVal.GetAsArray1D());
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
            //Utils.print1d(PreMergeMaxVal.GetAsArray1D());
            //Console.ReadLine();
            //Utils.print1d(TempBuffer.GetAsArray1D());

            refactorPart1Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);
            //accelerate.DefaultStream.Synchronize();
            //Utils.print1d(mergedBuffer.GetAsArray1D());
            //Utils.print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();

            // Console.WriteLine("Part 1:");
            // Console.WriteLine(watch.ElapsedMilliseconds);
            // Console.ReadLine();

            //Utils.print1d(mergedBuffer.GetAsArray1D());
            //Console.WriteLine("Testtiong");
            //Console.ReadLine();

            setBuffToValueDoubleKern(TempBuffer.Extent.ToIntIndex(), TempBuffer.View, 1.0);

            setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

            refactorPart2Phase1Kern(
                mergedBuffer.Extent.ToIntIndex(),
                mergedBuffer.View,
                TempBuffer.View,
                HolderBuffer.View
            );
            //Utils.print1d(TempBuffer.GetAsArray1D());
            //Utils.print1d(HolderBuffer.GetAsArray1D());
            //Console.ReadLine();

            refactorPart2Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);

            //Utils.print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();
            //AdjMergeBuffer.CopyFromCPU(refactorToMinimizeSize(mergedBuffer.GetAsArray1D()));

            //Utils.print1d(mergedBuffer.GetAsArray1D());
            //accelerate.DefaultStream.Synchronize();

            // Console.WriteLine("Refactor Arrs Part2:");

            //Console.WriteLine(watch.ElapsedMilliseconds);
            InitMaxMinKern(MergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            //Utils.print1d(MergeMaxVal.GetAsArray1D());

            // Utils.print1d(MergeMaxVal.GetAsArray1D());
            // Utils.print1d(MergeMinVal.GetAsArray1D());
            // Utils.print1d(mergedBuffer.GetAsArray1D());
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

            //Utils.print1d(mergedBuffer.GetAsArray1D());
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

            // Utils.print1d(CondNormBuffer.GetAsArray1D());
            // Utils.print1d(MergeNormBuffer.GetAsArray1D());

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
            //Utils.print1d(MergeNormBuffer.GetAsArray1D());
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

            double ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);

            double ent2 = EntropyBuffer2.GetAsArray1D()[0] / Math.Log(Constants.LOG_BASE);

            return ent1 - ent2;
        }

        // public double ConditiponalMIHolder(double[] firstVector, double[] secondVector, double[] conditionVector)
        // {
        //     double[] mergedVector = new double[firstVector.GetLength(0)];

        //     mergeArrays(firstVector, conditionVector, ref mergedVector);

        //     //Secondvector = first vector
        //     // conditionvector = second vector
        //     double firstCondEnt = calculateConditionalEntropy(secondVector, conditionVector);
        //     //Utils.print1d(refactorToMinimizeSize(mergedVector));

        //     //second vector = first vector
        //     // merged vector = second vector
        //     double secondCondEnt = calculateConditionalEntropy(secondVector, refactorToMinimizeSize(mergedVector));

        //     return firstCondEnt - secondCondEnt;
        // }
        #endregion

        #region Kernels
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
        protected static void mergeArraysAdjKernel(
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

        ///<summary>Sets every element in buff to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        protected static void setBuffToValueKernal(Index1D index, ArrayView1D<int, Stride1D.Dense> buff, int setvalue)
        {
            buff[index] = setvalue;
        }

        protected static void refactorKernal(Index1D index, ArrayView1D<double, Stride1D.Dense> inputView, int length)
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

protected static void refactorPart1Phase1Kernal(
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

        protected static void refactorPart1Phase2Kernal(
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

        protected static void refactorPart2Phase1Kernal(
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

        protected static void countnonNaN(
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

        protected static void refactorPart2Phase2Kernal(
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

        #endregion
    }
}
