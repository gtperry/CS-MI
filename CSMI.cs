using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;

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
	public class MI{
        const double LOG_BASE = 2.0;
        const int MAX_YDIM = 65535;
        const int GRIDSIZE = 1024;
        Device dev;
        //Accelerator accelerate;

        //Will be used later when optimized for GPU use
        Context context;

        public MI()
        {
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
            //MI m = new MI();
            this.dev = this.context.GetPreferredDevice(preferCPU: false);
        }
        public double calculateEntropy(double[] dataVector){
            

            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var MVBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            var MVNormBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            //var FreqBuffer = accelerate.Allocate1D<double>(new Index1D(dataVector.GetLength(0)));
            MVBuffer.CopyFromCPU(dataVector);
            var MaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var MinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjKernel);

            
            var CalcEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, int>(
                CalcEntropyKernel);

            var normalizeArrayKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(normalizeArrayKernel);

            InitMaxMinKern(MaxVal.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            GetMaxValKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            normalizeArrayKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MVNormBuffer.View, MinVal);
            var FreqBuffer = accelerate.Allocate1D<double>(new Index1D(MaxVal.GetAsArray1D()[0] - MinVal.GetAsArray1D()[0] + 1));
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

            CalcEntropyKern(FreqBuffer.Extent.ToIntIndex(), FreqBuffer.View, EntropyBuffer.View, dataVector.GetLength(0));
            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;


        }
        public double mergeArrays(double[] firstVector, double[] secondVector, ref double[] outputVector){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            var outputBuffer = accelerate.Allocate1D<double>(new Index1D(outputVector.GetLength(0)));

       
            
            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var StateCount = accelerate.Allocate1D<int>(new Index1D(1));

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            var normalizeArrayKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(normalizeArrayKernel);
            var mergeArraysKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                int, int>(mergeArraysKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);

            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];

            var StateMap = accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1)*(secondnumstates+ 1)));
            //outputVector = outputBuffer.GetAsArray1D();
            mergeArraysKern(StateCount.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, StateMap.View, outputBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);
            outputVector = outputBuffer.GetAsArray1D();
            // print1d(outputVector);
            double answer = StateCount.GetAsArray1D()[0];
            accelerate.Dispose();
            return answer;




        }
        public double calculateJointEntropy(double[] firstVector, double[] secondVector){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            var TestBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            var normalizeArrayKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(normalizeArrayKernel);

            var BuildJointFreqKern= accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double>(setBuffToValue2DKernal);

            var CalcJointEntropyKern = accelerate.LoadAutoGroupedStreamKernel<Index2D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                int>(CalcJointEntropyKernel);

            var IndexedCalcJointEntropyKern = accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(IndexedCalcJointEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjustedKernel);
            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            //Console.WriteLine(firstnumstates);
            //Console.WriteLine(secondnumstates);


            var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstnumstates + 1, secondnumstates+ 1));


            
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);

            BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);

            //CalcJointEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, EntropyBuffer.View, firstVector.GetLength(0));
            IndexedCalcJointEntropyKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View,FirstNormBuffer.View, SecondNormBuffer.View, EntropyBuffer.View, firstVector.GetLength(0));

            // Console.WriteLine("joint");
            // Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine(TestBuffer.GetAsArray1D()[0]);

            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;

        }
        public double calculateConditionalEntropy(double[] firstVector, double[] secondVector){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            var TestBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            var normalizeArrayKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(normalizeArrayKernel);

            var BuildJointFreqKern= accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double>(setBuffToValue2DKernal);

            var CalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<Index2D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(CalcConditionalEntropyKernel);

            var IndexedCalcConditionalEntropyKern = accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(IndexedCalcConditionalEntropyKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            Console.WriteLine(firstnumstates);
            Console.WriteLine(secondnumstates);


            var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstnumstates +1, secondnumstates+1));


            var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);
            //print2d(JointBuffer.GetAsArray2D());
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine("JOINT");
            //print2d(JointBuffer.GetAsArray2D());
            // Console.WriteLine("SecondCountMap");
            //print1d(SecondCountMap.GetAsArray1D());
            // /CalcConditionalEntropyKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));

            IndexedCalcConditionalEntropyKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, FirstNormBuffer.View, SecondNormBuffer.View,EntropyBuffer.View, firstVector.GetLength(0));

            // Console.WriteLine("TESTTT");
            // Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            // Console.WriteLine(TestBuffer.GetAsArray1D()[0]);


            // Console.WriteLine("ENTROPY ARR");
            // print1d(EntropyBuffer.GetAsArray1D());
            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;


        }
        public double calculateMutualInformation(double[] firstVector, double[] secondVector){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);

            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            

            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            var normalizeArrayKern= accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(normalizeArrayKernel);

            var BuildJointFreqKern= accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double>(setBuffToValue2DKernal);

            var CalcMIKern = accelerate.LoadAutoGroupedStreamKernel<Index2D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(CalcMIKernel);

            var IndexedCalcMIKern = accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView2D<double, Stride2D.DenseX>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(IndexedCalcMIKernel);

            var setBuffToValueDoubleKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueDoubleKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjustedKernel);

            InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
            //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
            InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            //Console.WriteLine(firstnumstates);
            //Console.WriteLine(secondnumstates);


            var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstnumstates + 1, secondnumstates + 1));


            var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);

            BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);
            BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
            BuildFreqKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstCountMap.View);

            
            //CalcMIKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));

            IndexedCalcMIKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View,FirstNormBuffer.View, SecondNormBuffer.View, EntropyBuffer.View, firstVector.GetLength(0));


            
            
            double answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            accelerate.Dispose();
            return answer;


        }
        public double calculateConditionalMutualInformation(double[] firstVector, double[] secondVector, double[] conditionVector){
            double[] mergedVector = new double[firstVector.GetLength(0)];

            mergeArrays(firstVector, conditionVector, ref mergedVector);
            

            double firstCondEnt = calculateConditionalEntropy(secondVector, conditionVector);
            double secondCondEnt = calculateConditionalEntropy(secondVector, mergedVector);

            return firstCondEnt - secondCondEnt;

        }
        static void mergeArraysKernel(Index1D index,
            ArrayView1D<double, Stride1D.Dense> FirstNormView,
            ArrayView1D<double, Stride1D.Dense> SecondNormView,
            ArrayView1D<int, Stride1D.Dense> StateMap,
            ArrayView1D<double, Stride1D.Dense> OutputView,
            ArrayView1D<int, Stride1D.Dense> SCount,
            int firstnumstates,
            int length
            ){
            int curindex;
            int statecount = 1;
            for(int i = 0; i < length; i++){
                curindex = (int)FirstNormView[new Index1D(i)] + ((int)SecondNormView[new Index1D(i)] * firstnumstates);
                if(StateMap[new Index1D(curindex)] == 0){
                    StateMap[new Index1D(curindex)] = statecount;
                    statecount +=1;
                }
                OutputView[new Index1D(i)] = (double)StateMap[curindex];
            }
            SCount[index] = statecount;

        }
        static void CalcEntropyKernel(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> FreqView, 
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            if(FreqView[index] > 0){
                val = -1 * (FreqView[index]/(double)length) * Math.Log(FreqView[index]/(double)length);
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }
        }
        static void CalcConditionalEntropyKernel(Index2D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> condView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            if(FreqView[index] > 0 && condView[index.Y] > 0){
                val = -1 * ((FreqView[index]/(double)length) * Math.Log( (FreqView[index]/(double)length) /  (condView[index.Y]/(double)length)));
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }
        static void IndexedCalcConditionalEntropyKernel(Index1D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> condView,
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            Index2D ind2 = new Index2D((int)FirstView[index], (int)SecondView[index]);
            if(FreqView[ind2] > 0 && condView[ind2.Y] > 0){
                val = -1 * ((FreqView[ind2]/(double)length) * Math.Log( (FreqView[ind2]/(double)length) /  (condView[ind2.Y]/(double)length)));
                val = val / FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }
        static void CalcMIKernel(Index2D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            if(FreqView[index] > 0 && first[index.X] > 0 && second[index.Y] > 0){
                val = ((FreqView[index]/(double)length) * Math.Log( ((FreqView[index]/(double)length) /  (first[index.X]/(double)length))/ (second[index.Y]/(double)length)  ));
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }
        static void IndexedCalcMIKernel(Index1D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            Index2D ind2 = new Index2D((int)FirstView[index],(int)SecondView[index]);
            if(FreqView[ind2] > 0 && first[(int)FirstView[index]] > 0 && second[(int)SecondView[index]] > 0){
                val = ((FreqView[ind2]/(double)length) * Math.Log( ((FreqView[ind2]/(double)length) /  (first[ind2.X]/(double)length))/ (second[ind2.Y]/(double)length)  ));
                val = val / FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }
        static void CalcJointEntropyKernel(Index2D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            if(FreqView[index] > 0){
                val = -1 * (FreqView[index]/(double)length) * Math.Log( (FreqView[index]/(double)length) );
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }

        static void IndexedCalcJointEntropyKernel(Index1D index, 
            ArrayView2D<double, Stride2D.DenseX> FreqView, 
            ArrayView1D<double, Stride1D.Dense> FirstView,
            ArrayView1D<double, Stride1D.Dense> SecondView,
            ArrayView1D<double, Stride1D.Dense> entropy,
            int length){
            double val;
            Index2D ind2 = new Index2D((int)FirstView[index],(int)SecondView[index]);
            if(FreqView[ind2] > 0){
                val = -1 * (FreqView[ind2]/(double)length) * Math.Log( (FreqView[ind2]/(double)length) );
                val = val/FreqView[ind2];
                Atomic.Add(ref entropy[new Index1D(0)], val);
            }

        }
		static void normalizeArrayKernel(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> outputView,
            ArrayView1D<int, Stride1D.Dense> minVal)
        {
            outputView[index] =  Math.Floor(inputView[index]) - (double)minVal[new Index1D(0)];

        }
        static void GetMaxMinValKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal)
        {
            Atomic.Max(ref MaxVal[new Index1D(0)],  (int)Math.Floor(aView[index]));
            Atomic.Min(ref MinVal[new Index1D(0)],  (int)Math.Floor(aView[index]));
            

        }
        static void InsertBufferIntoTree(Index1D index,
            ArrayView1D<int, Stride1D.Dense> View,
            ArrayView1D<int, Stride1D.Dense> Tree){

        }
        static void InitMaxMinKernel(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal)
        {
            
            MaxVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
            MinVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
        }

        static void setBuffToValueKernal(Index1D index, 
            ArrayView1D<int, Stride1D.Dense> buff, 
            int setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setBuffToValueDoubleKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setBuffToValue2DKernal(Index2D index, 
            ArrayView2D<double, Stride2D.DenseX> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void BuildFreqKernel(Index2D index, 
            ArrayView1D<double, Stride1D.Dense> input, 
            ArrayView1D<double, Stride1D.Dense> output)
        {
            
            if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y])){
                Atomic.Add(ref output[index.X], 1.0);
            }
            
        }
        static void BuildFreqAdjKernel(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> input, 
            ArrayView1D<double, Stride1D.Dense> output)
        {
            
            //if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y])){
            Atomic.Add(ref output[(int)Math.Floor(input[index])], 1.0);
            //}
            
        }

        static void LargeBuildFreqKernel(Index3D index, 
            ArrayView1D<double, Stride1D.Dense> input, 
            ArrayView1D<double, Stride1D.Dense> output,
            int gridsize, int length)
        {
            if((index.Z * gridsize) + index.Y < length){
                if(Math.Floor(input[index.X]) == Math.Floor(input[(index.Z * gridsize) + index.Y])){
                    Atomic.Add(ref output[index.X], 1.0);
                }
            }
            
            
        }
        static void BuildFreqAdjustedKernel(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> input, 
            ArrayView1D<double, Stride1D.Dense> output)
        {
            
            //if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y]) && Math.Floor(input[index.X]) != 0){
            Atomic.Add(ref output[new Index1D((int)(input[index]))], 1.0);
            //}
            
        }
        static void BuildJointFreqKernel(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> first, 
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView2D<double, Stride2D.DenseX> output)
        {
            
            //if(Math.Floor(first[index.X]) == Math.Floor(first[index.Y]) && Math.Floor(second[index.X]) == Math.Floor(second[index.Y])){
            Atomic.Add(ref output[new Index2D((int)first[index], (int)second[index])], 1.0);
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
        void test(int length){
            Stopwatch stop = new Stopwatch();
            Console.Write("LENGTH =");
            Console.WriteLine(length);
            Random rand = new Random();
            double[] a = new double[length];
            for (int i = 0; i < length; i++) {
              a[i] = rand.NextDouble() * 10000;
            }
            double[] b = new double[length];
            for (int i = 0; i < length; i++) {
              b[i] = rand.NextDouble() * 100000;
            }
            double[] c = new double[length];
            for (int i = 0; i < length; i++) {
              c[i] = rand.NextDouble() * 10000;
            }
            stop.Start();
            calculateMutualInformation(a,b);
            stop.Stop();
            Console.Write("Elapsed time for Mutual Information: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            calculateEntropy(a);
            stop.Stop();
            Console.Write("Elapsed time for Calculate Entropy: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            calculateConditionalEntropy(a,b);
            stop.Stop();
            Console.Write("Elapsed time for Calculate Conditional Entropy: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            calculateJointEntropy(a,b);
            stop.Stop();
            Console.Write("Elapsed time for Calculate Joint Entropy: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            //calculateConditionalMutualInformation(a,b,c);
            stop.Stop();
            Console.Write("Elapsed time for Conditional Mutual Information: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            Console.WriteLine();
            Console.WriteLine("----------------------------");
            Console.WriteLine();
        }
		static void Main(string[] args)
	    {
	    	double[] a = new[] {4.2, 5.43, 3.221, 7.34235, 1.931, 1.2, 5.43, 8.0, 7.34235, 1.931};
            double[] b = new[] {2.2, 3.43, 1.221, 9.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 4.931};
            double[] c = new double [10];
	    	Context context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
            MI m = new MI();
            Device dev = context.GetPreferredDevice(preferCPU: false);

            Accelerator accelerate = dev.CreateAccelerator(context);
	    	var MVBuffer = accelerate.Allocate1D<double>(new Index1D(10));
            var FreqBuffer = accelerate.Allocate1D<double>(new Index1D(10));
	    	MVBuffer.CopyFromCPU(a);
	    	var MaxVal = accelerate.Allocate1D<int>(new Index1D(1));
	    	var MinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(10));
	    	var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(GetMaxMinValKernal);
            var InitMaxMinKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(InitMaxMinKernel);
            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setBuffToValueKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqKernel);

            var CalcEntropyKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, int>(
                CalcEntropyKernel);

            // for (int i = 1; i < 11; i++){
            //   m.test((int)Math.Pow(10,i));
            // }
            // InitMaxMinKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            // Console.WriteLine(MaxVal.GetAsArray1D()[0]);
            // Console.WriteLine(MinVal.GetAsArray1D()[0]);
            // GetMaxValKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);

            // Console.WriteLine(MaxVal.GetAsArray1D()[0]);
            // Console.WriteLine(MinVal.GetAsArray1D()[0]);
            // BuildFreqKern(new Index2D(MVBuffer.Extent.ToIntIndex().X,MVBuffer.Extent.ToIntIndex().X), MVBuffer.View, FreqBuffer.View);
            // m.print1d(MVBuffer.GetAsArray1D());
            // m.print1d(FreqBuffer.GetAsArray1D());

            // CalcEntropyKern(FreqBuffer.Extent.ToIntIndex(), FreqBuffer.View, EntropyBuffer.View, 10);
            // m.print1d(EntropyBuffer.GetAsArray1D());
            Console.WriteLine(m.calculateMutualInformation(a,b));
            Console.WriteLine("MI ^^^");

            Console.WriteLine(m.calculateConditionalEntropy(a,b));
            Console.WriteLine("Conditional Entropy ^^^");
            // /Console.ReadLine();
            Console.WriteLine(m.calculateJointEntropy(a,b));
            Console.WriteLine("JOINT ENTROPY^^^");
            Console.WriteLine(m.calculateEntropy(b));

            m.mergeArrays(a,b,ref c);
            Console.WriteLine(m.calculateEntropy(c));
            //m.print1d(c);
	    	Console.WriteLine("Hello World");
	    }
	        
	}
}		