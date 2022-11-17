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

using Microsoft.Data.Analysis;

namespace CSMI
{
    public class OutputGroup{
        string id1;
        string id2;
        double MIVal;
        public OutputGroup(string id1, string id2, double MI){
            this.id1 = id1;
            this.id2 = id2;
            this.MIVal = MI;
        }
        public override string ToString()
        {
            return "(" + this.id1 + ", " + this.id2 + "): " + this.MIVal;
        }
    }
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
        public double[] refactorArray(double[] arr){
            IDictionary<int, double> newarr = new Dictionary<int, double>();
            double[] refactoredarr = new double[arr.GetLength(0)];
            double count = 0.0;
            int current;
            for(int i = 0; i < arr.GetLength(0); i++){
                current= (int)Math.Floor(arr[i]);
                if (!newarr.ContainsKey(current)){
                    newarr.Add(current, count);
                    count+=1.0;
                }
                refactoredarr[i] = newarr[current];

            
            }
            //print1d(refactoredarr);
            return refactoredarr;
        }
        public double[] refactorToMinimizeSize(double[] arr){
            IDictionary<int, double> newarr = new Dictionary<int, double>();
            IDictionary<int, double> freqarr = new Dictionary<int, double>();
            double[] refactoredarr = new double[arr.GetLength(0)];
            double count = 0.0;
            int current;
            double temp;
            bool allnan = true;
            for(int i = 0; i < arr.GetLength(0); i++){
                current= (int)Math.Floor(arr[i]);
                if (!freqarr.ContainsKey(current)){
                    freqarr.Add(current, 1.0);
                    
                }
                else{
                    temp = freqarr[current];
                    freqarr.Remove(current);
                    freqarr.Add(current, temp+1.0);

                }
               //refactoredarr[i] = newarr[current];

            
            }
            for(int i = 0; i < arr.GetLength(0); i++){
                current= (int)Math.Floor(arr[i]);
                if (!newarr.ContainsKey(current)){
                    if(freqarr[current] > 1.0){
                        newarr.Add(current, count);
                        count+=1.0;
                        allnan = false;
                    }
                    else{
                        newarr.Add(current, Double.NaN);

                    }
                    
                }
                refactoredarr[i] = newarr[current];

            
            }
            if(Double.IsNaN(refactoredarr[0])){
                refactoredarr[0] = count + 1;
            }
            //Console.WriteLine("Refactored arr");
            //print1d(arr);
            //print1d(refactoredarr);
            //Console.ReadLine();
            return refactoredarr;

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
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            double answer;
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
                ArrayView1D<int, Stride1D.Dense>>(TestGetMaxMinValKernal);
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
            //print1d(SecondBuffer.GetAsArray1D());
            GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
            //print1d(SecondMinVal.GetAsArray1D());

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
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            Console.WriteLine("First and second norm:");
            //print1d(SecondMinVal.GetAsArray1D());
            //print1d(SecondNormBuffer.GetAsArray1D());
            //Console.ReadLine();

            BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);
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
            IndexedCalcConditionalEntropyKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, FirstNormBuffer.View, SecondNormBuffer.View,EntropyBuffer.View, firstVector.GetLength(0));

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
        
        public double calculateConditionalEntropyAdjusted(double[] firstVector, double[] secondVector, int nanvals){
            //// NOTES: Need to account for the NaN values, have them add a 1/n addition to the entropy
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            double answer;
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
                ArrayView1D<int, Stride1D.Dense>>(TestGetMaxMinValKernal);
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
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
            // Console.WriteLine("Norms");
            // print1d(FirstNormBuffer.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            int firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
            int secondnumstates = SecondMaxVal.GetAsArray1D()[0];
            // Console.WriteLine(firstnumstates);
            // Console.WriteLine(secondnumstates);


            var JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstnumstates +1, secondnumstates+1));
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            var FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
            var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
            //Console.WriteLine(EntropyBuffer.GetAsArray1D()[0]);
            //answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
            // Console.WriteLine("First and second norm:");
            // print1d(SecondMinVal.GetAsArray1D());
            // print1d(SecondNormBuffer.GetAsArray1D());
            // Console.ReadLine();

            BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);
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
            IndexedCalcConditionalEntropyKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View, SecondCountMap.View, FirstNormBuffer.View, SecondNormBuffer.View,EntropyBuffer.View, firstVector.GetLength(0));

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


        public List<double> MulticalculateMutualInformation(List<double[]> arr){
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));


            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));
            var EntropyBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            var JointBuffer  = accelerate.Allocate2DDenseX<double>(new Index2D(1,1));
            var FirstCountMap  = accelerate.Allocate1D<double>(new Index1D(1));
            var SecondCountMap  = accelerate.Allocate1D<double>(new Index1D(1));
            int firstnumstates;
            int secondnumstates;
            double answer;

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
            List<double> MIanswers = new List<double>();
            Stopwatch watch = new Stopwatch();
            for(int i = 0; i < arr.Count; i++){
                watch.Start();
                for(int j = 0; j < arr.Count; j++){
                    
                    if(i !=j){

                        FirstBuffer.CopyFromCPU(arr[i]);
                        SecondBuffer.CopyFromCPU(arr[j]);
                        InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
                        GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
                        //Console.WriteLine(FirstMaxVal.GetAsArray1D()[0]);
                        InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
                        GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);

                        normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal);
                        normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal);
                        
                        firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
                        secondnumstates = SecondMaxVal.GetAsArray1D()[0];
                        //Console.WriteLine(firstnumstates);
                        //Console.WriteLine(secondnumstates);


                        JointBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(firstnumstates + 1, secondnumstates + 1));


                        FirstCountMap = accelerate.Allocate1D<double>(new Index1D(firstnumstates));
                        SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
                        //setBuffToValue2DKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, 0.0);
                        // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
                        // setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
                        setBuffToValueDoubleKern(FirstCountMap.Extent.ToIntIndex(), FirstCountMap.View, 0.0);
                        setBuffToValueDoubleKern(FirstCountMap.Extent.ToIntIndex(), FirstCountMap.View, 0.0);
                        BuildJointFreqKern(SecondNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, JointBuffer.View);
                        BuildFreqKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondCountMap.View);
                        BuildFreqKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstCountMap.View);

                        
                        //CalcMIKern(JointBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View, EntropyBuffer.View, firstVector.GetLength(0));
                        setBuffToValueDoubleKern(EntropyBuffer.Extent.ToIntIndex(), EntropyBuffer.View, 0.0);
                        IndexedCalcMIKern(SecondNormBuffer.Extent.ToIntIndex(), JointBuffer.View, FirstCountMap.View,SecondCountMap.View,FirstNormBuffer.View, SecondNormBuffer.View, EntropyBuffer.View, arr[0].GetLength(0));


                        
                        
                        answer = EntropyBuffer.GetAsArray1D()[0] / Math.Log(LOG_BASE);
                        MIanswers.Add(answer);
                    }

                }
                watch.Stop();
                Console.WriteLine(watch.ElapsedMilliseconds);
                watch.Reset();
            }
            

            
            accelerate.Dispose();
            return MIanswers;


        }

        public OutputGroup[] findBestCondMutInf(double[] targetvar, List<double[]> arr, string[] columnnames){

            Stopwatch watch = new Stopwatch();
            watch.Start();
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(targetvar.GetLength(0)));
            var CondBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            //var testIndex = new LongIndex2D(10000, 1000000);
            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(targetvar.GetLength(0)));
            var CondNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));

            var mergedBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));
            //var AdjMergeBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));

            var MergeNormBuffer = accelerate.Allocate1D<double>(new Index1D(arr[0].GetLength(0)));

            //FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(targetvar);
            //CondBuffer.CopyFromCPU(conditionVector);

            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var CondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var CondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var MergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var MergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var PreMergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var PreMergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var EntropyBuffer1 = accelerate.Allocate1D<double>(new Index1D(1));
            var EntropyBuffer2 = accelerate.Allocate1D<double>(new Index1D(1));

            var StateCount = accelerate.Allocate1D<int>(new Index1D(1));

            var CountNonNaN = accelerate.Allocate1D<double>(new Index1D(1));
            var HolderBuffer = accelerate.Allocate1D<double>(new Index1D(1));


            int firstnumstates;
            int secondnumstates;

            int condnumstates;

            int mergenumstates;
            double ent1;
            double ent2;

            int premergenumstates;
            var StateMap = accelerate.Allocate2DDenseX<int>(new Index2D(1, 1));

            var JointBuffer1 = accelerate.Allocate2DDenseX<double>(new Index2D(1, 1));
            var JointBuffer2 = accelerate.Allocate2DDenseX<double>(new Index2D(1, 1));

                        //EntropyBuffer1.GetAsArray1D();

                        
                        //var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            var CondCountMap = accelerate.Allocate1D<double>(new Index1D(1));
            var MergeCountMap = accelerate.Allocate1D<double>(new Index1D(1));

            var TempBuffer = accelerate.Allocate1D<double>(new Index1D(1));
            var TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(1));



            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(TestGetMaxMinValKernal);
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
                ArrayView2D<int, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                int, int>(mergeArraysAdjKernel);
            var BuildJointFreqKern= accelerate.LoadAutoGroupedStreamKernel<Index1D, 
                ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(BuildJointFreqKernel);

            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double>(setBuffToValue2DKernal);

            var setBuffToValue2DIntKern = accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<int, Stride2D.DenseX>,
                int>(setBuffToValue2DIntKernal);

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

            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setBuffToValueKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjKernel);

        

            var RefactorPart1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                int>(
                refactorKernal);

            var refactorPart1Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                refactorPart1Phase1Kernal);

            var refactorPart1Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                refactorPart1Phase2Kernal);


            var refactorPart2Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
                >(
                refactorPart2Phase1Kernal);

            var refactorPart2Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
                >(
                refactorPart2Phase2Kernal);

            var countnonNaNKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(countnonNaN);

            List<OutputGroup> output = new List<OutputGroup>();

            for(int i = 0; i < columnnames.GetLength(0)-1; i++){
                setBuffToValueDoubleKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, 0.0);
                Console.WriteLine(columnnames[i]);
                Console.ReadLine();
                FirstBuffer.CopyFromCPU(arr[i]);

                for(int j = 0; j < columnnames.GetLength(0)-1; j++){
                    Console.Write("|");
                    if(!( i == j)){

                        setBuffToValueDoubleKern(CondBuffer.Extent.ToIntIndex(), CondBuffer.View, 0.0);

                        CondBuffer.CopyFromCPU(arr[j]);
                        
                        setBuffToValueKern(FirstMaxVal.Extent.ToIntIndex(), FirstMaxVal.View, 0);
                        setBuffToValueKern(FirstMinVal.Extent.ToIntIndex(), FirstMinVal.View, 0);

                        setBuffToValueKern(SecondMaxVal.Extent.ToIntIndex(), SecondMaxVal.View, 0);
                        setBuffToValueKern(SecondMinVal.Extent.ToIntIndex(), SecondMinVal.View, 0);

                        setBuffToValueKern(CondMaxVal.Extent.ToIntIndex(), CondMaxVal.View, 0);
                        setBuffToValueKern(CondMinVal.Extent.ToIntIndex(), CondMinVal.View, 0);

                        setBuffToValueKern(PreMergeMaxVal.Extent.ToIntIndex(), PreMergeMaxVal.View, 0);
                        setBuffToValueKern(PreMergeMinVal.Extent.ToIntIndex(), PreMergeMinVal.View, 0);

                        setBuffToValueKern(MergeMaxVal.Extent.ToIntIndex(), MergeMaxVal.View, 0);
                        setBuffToValueKern(MergeMinVal.Extent.ToIntIndex(), MergeMinVal.View, 0);

                        setBuffToValueDoubleKern(EntropyBuffer1.Extent.ToIntIndex(), EntropyBuffer1.View, 0.0);
                        setBuffToValueDoubleKern(EntropyBuffer2.Extent.ToIntIndex(), EntropyBuffer2.View, 0.0);

                        setBuffToValueDoubleKern(CountNonNaN.Extent.ToIntIndex(), CountNonNaN.View, 0.0);
                        setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 0.0);
                        setBuffToValueDoubleKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, 0.0);


                        setBuffToValueDoubleKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, 0.0);

                        setBuffToValueDoubleKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, 0.0);

                        setBuffToValueDoubleKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, 0.0);




                        setBuffToValueKern(StateCount.Extent.ToIntIndex(), StateCount.View, 0);






                        InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);
                        GetMaxValKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstMaxVal.View, FirstMinVal.View);

                        InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
                        GetMaxValKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondMaxVal.View, SecondMinVal.View);
                        //print1d(CondBuffer.GetAsArray1D());
                        InitMaxMinKern(CondMaxVal.Extent.ToIntIndex(), CondBuffer.View, CondMaxVal.View, CondMinVal.View);
                        GetMaxValKern(CondBuffer.Extent.ToIntIndex(), CondBuffer.View, CondMaxVal.View, CondMinVal.View);

                        normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal.View);
                        normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal.View);
                        normalizeArrayKern(CondBuffer.Extent.ToIntIndex(), CondBuffer.View, CondNormBuffer.View, CondMinVal.View);
                        // accelerate.DefaultStream.Synchronize();

                        // watch.Stop();
                        // //RefactorPart1TestKern(testIndex, FirstBuffer.View, SecondBuffer.View);
                        // Console.WriteLine("MaxMin ETC: ");

                        // Console.WriteLine(watch.ElapsedMilliseconds);
                        // watch.Reset();

                        // accelerate.DefaultStream.Synchronize();
                        // watch.Start();

                        firstnumstates = FirstMaxVal.GetAsArray1D()[0];           
                        secondnumstates = SecondMaxVal.GetAsArray1D()[0];
                        condnumstates = CondMaxVal.GetAsArray1D()[0];

                        Console.WriteLine("NumStates");
                        Console.WriteLine(firstnumstates);
                        Console.WriteLine(secondnumstates);
                        Console.WriteLine(condnumstates);

                        StateMap.Dispose();
                        StateMap = accelerate.Allocate2DDenseX<int>(new Index2D(firstnumstates + 1, secondnumstates+ 1)); // accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1)*(secondnumstates+ 1)));
                        //outputVector = mergedBuffer.GetAsArray1D();
                        setBuffToValue2DIntKern(StateMap.Extent.ToIntIndex(), StateMap.View, 0);
                        setBuffToValueKern(StateCount.Extent.ToIntIndex(), StateCount.View, 1);
                        //print1d(FirstNormBuffer.GetAsArray1D());
                        premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];

                        //print1d(SecondNormBuffer.GetAsArray1D());
                        //Console.ReadLine();
                        Console.WriteLine(firstnumstates);
                        Console.WriteLine(secondnumstates);
                        //print1d(FirstNormBuffer.GetAsArray1D());
                        Console.WriteLine("WHy");

                        //print1d(SecondNormBuffer.GetAsArray1D());
                        
                        mergeArraysKern(mergedBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, CondNormBuffer.View, StateMap.View, mergedBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);
                       // mergeArraysKern(StateCount.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, StateMap.View, mergedBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);

                        //// TO SHRINK INPUTS
                        // TempBuffer.Dispose();

                        // TempBuffer = accelerate.Allocate1D<double>(new Index1D(firstnumstates +1));
                        // setBuffToValueDoubleKern(TempBuffer.Extent.ToIntIndex(), TempBuffer.View, 0.0);

                        // TempBuffer2.Dispose();

                        // TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(firstnumstates +1));
                        // Console.WriteLine("Pre");
                        // Console.WriteLine("FirstNorm");
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer");
                        // print1d(TempBuffer.GetAsArray1D());
                        // refactorPart1Phase1Kern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, TempBuffer.View);
                        // Console.WriteLine("P1PHASE1");

                        // Console.WriteLine("FirstNorm");
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer2");
                        // print1d(TempBuffer.GetAsArray1D());
                        

                        // refactorPart1Phase2Kern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, TempBuffer.View);
                        // Console.WriteLine("P1PHASE2");

                        // Console.WriteLine("FirstNorm");
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer");
                        // print1d(TempBuffer.GetAsArray1D());
                        
                        // setBuffToValueDoubleKern(TempBuffer2.Extent.ToIntIndex(), TempBuffer2.View, 1.0);

                        // setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

                        // refactorPart2Phase1Kern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, TempBuffer.View, TempBuffer2.View, HolderBuffer.View);
                        // Console.WriteLine("P2PHASE1");

                        // Console.WriteLine("FirstNorm");
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer");
                        // print1d(TempBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer2");
                        // print1d(TempBuffer2.GetAsArray1D());
                        // Console.WriteLine("HolderBuffer");
                        // print1d(HolderBuffer.GetAsArray1D());
                        
                        

                        // refactorPart2Phase2Kern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, TempBuffer.View);
                        // Console.WriteLine("P2PHASE2");

                        // Console.WriteLine("FirstNorm");
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        // Console.WriteLine("TempBuffer2");
                        // print1d(TempBuffer.GetAsArray1D());
                        // Console.WriteLine("HolderBuffer");
                        // print1d(HolderBuffer.GetAsArray1D());

                        // setBuffToValueKern(FirstMaxVal.Extent.ToIntIndex(), FirstMaxVal.View, 0);
                        // setBuffToValueKern(FirstMinVal.Extent.ToIntIndex(), FirstMinVal.View, 0);
                        // InitMaxMinKern(FirstMaxVal.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
                        // GetMaxValKern(FirstNormBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, FirstMaxVal.View, FirstMinVal.View);
                        // firstnumstates = FirstMaxVal.GetAsArray1D()[0];           

// ///////
//                         TempBuffer.Dispose();

//                         TempBuffer = accelerate.Allocate1D<double>(new Index1D(secondnumstates +1));
//                         TempBuffer2.Dispose();

//                         TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(secondnumstates +1));
                       
//                         refactorPart1Phase1Kern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, TempBuffer.View);
                        

//                         refactorPart1Phase2Kern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, TempBuffer.View);
                        
//                         setBuffToValueDoubleKern(TempBuffer2.Extent.ToIntIndex(), TempBuffer2.View, 1.0);

//                         setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

//                         refactorPart2Phase1Kern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, TempBuffer.View, TempBuffer2.View, HolderBuffer.View);
                        

//                         refactorPart2Phase2Kern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, TempBuffer.View);

//                         setBuffToValueKern(SecondMaxVal.Extent.ToIntIndex(), SecondMaxVal.View, 0);
//                         setBuffToValueKern(SecondMinVal.Extent.ToIntIndex(), SecondMinVal.View, 0);
//                         InitMaxMinKern(SecondMaxVal.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
//                         GetMaxValKern(SecondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, SecondMaxVal.View, SecondMinVal.View);
//                         secondnumstates = SecondMaxVal.GetAsArray1D()[0];           

//////////////

                        // TempBuffer.Dispose();

                        // TempBuffer = accelerate.Allocate1D<double>(new Index1D(condnumstates +1));
                        // TempBuffer2.Dispose();

                        // TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(condnumstates +1));
                       
                        // refactorPart1Phase1Kern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, TempBuffer.View);
                        

                        // refactorPart1Phase2Kern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, TempBuffer.View);
                        
                        // setBuffToValueDoubleKern(TempBuffer2.Extent.ToIntIndex(), TempBuffer2.View, 1.0);

                        // setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

                        // refactorPart2Phase1Kern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, TempBuffer.View, TempBuffer2.View, HolderBuffer.View);
                        

                        // refactorPart2Phase2Kern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, TempBuffer.View);

                        // setBuffToValueKern(CondMaxVal.Extent.ToIntIndex(), CondMaxVal.View, 0);
                        // setBuffToValueKern(CondMinVal.Extent.ToIntIndex(), CondMinVal.View, 0);
                        // InitMaxMinKern(CondMaxVal.Extent.ToIntIndex(), CondNormBuffer.View, CondMaxVal.View, CondMinVal.View);
                        // GetMaxValKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, CondMaxVal.View, CondMinVal.View);
                        // condnumstates = CondMaxVal.GetAsArray1D()[0];         



///////// TO SHRINK INPUTS


                        //premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];
                        print1d(StateCount.GetAsArray1D());
                        Console.WriteLine("What");
                        StateMap.Dispose();

                        //outputVector = mergedBuffer.GetAsArray1D();
                        // print1d(outputVector);
                        // Console.WriteLine("InHere");
                        // print1d(SecondNormBuffer.GetAsArray1D());
                        // print1d(FirstNormBuffer.GetAsArray1D());
                        //print1d(mergedBuffer.GetAsArray1D());
                        // Console.ReadLine();
                        InitMaxMinKern(PreMergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, PreMergeMaxVal.View, PreMergeMinVal.View);
                        //premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];

                        GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, PreMergeMaxVal.View, PreMergeMinVal.View);
                        
                        premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];
                        // if(premergenumstates > 10000){
                        //     Console.WriteLine("Bug tedsting");
                        //     Console.WriteLine(i);
                        //     Console.WriteLine(j);
                        //     print1d(mergedBuffer.GetAsArray1D());
                        //     print1d(FirstNormBuffer.GetAsArray1D());

                        //     print1d(SecondNormBuffer.GetAsArray1D());

                        //     Console.ReadLine();
                        //     //1079803904
                        // }
                        TempBuffer.Dispose();

                        TempBuffer = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));
                        TempBuffer2.Dispose();

                        TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));
                       
                        refactorPart1Phase1Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);
                        

                        refactorPart1Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);
                        
                        setBuffToValueDoubleKern(TempBuffer2.Extent.ToIntIndex(), TempBuffer2.View, 1.0);

                        setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 1.0);

                        refactorPart2Phase1Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View, TempBuffer2.View, HolderBuffer.View);
                        

                        refactorPart2Phase2Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View);

                        //print1d(mergedBuffer.GetAsArray1D());
                        //Console.ReadLine();
                        //AdjMergeBuffer.CopyFromCPU(refactorToMinimizeSize(mergedBuffer.GetAsArray1D()));

                        //print1d(mergedBuffer.GetAsArray1D());
                        //accelerate.DefaultStream.Synchronize();

                        //watch.Stop();

                        //Console.WriteLine("Refactor Arrs Part2:");
                        ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);

                        //Console.WriteLine(watch.ElapsedMilliseconds);
                        //watch.Reset();
                        InitMaxMinKern(MergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
                        GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
                        //print1d(MergeMaxVal.GetAsArray1D());

                        // print1d(MergeMaxVal.GetAsArray1D());
                        // print1d(MergeMinVal.GetAsArray1D());
                        // print1d(mergedBuffer.GetAsArray1D());
                        // Console.ReadLine();

                        normalizeArrayKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeNormBuffer.View, MergeMinVal.View);

                        mergenumstates = MergeMaxVal.GetAsArray1D()[0];
                        Console.WriteLine("Merge sTates");
                        Console.WriteLine(mergenumstates);
                        Console.WriteLine(premergenumstates);


                        setBuffToValueDoubleKern(CountNonNaN.Extent.ToIntIndex(), CountNonNaN.View, 1.0);
                        countnonNaNKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, CountNonNaN.View);

                        ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);

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
                        JointBuffer1.Dispose();
                        JointBuffer2.Dispose();
                        CondCountMap.Dispose();
                        MergeCountMap.Dispose();
                        Console.WriteLine(firstnumstates);
                        Console.WriteLine(secondnumstates);
                        Console.WriteLine(condnumstates);
                        JointBuffer1 = accelerate.Allocate2DDenseX<double>(new Index2D(secondnumstates +1, condnumstates+1));
                        JointBuffer2 = accelerate.Allocate2DDenseX<double>(new Index2D(secondnumstates +1, mergenumstates+1));

                        //EntropyBuffer1.GetAsArray1D();

                        
                        //var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
                        //Console.WriteLine(condnumstates);
                        CondCountMap = accelerate.Allocate1D<double>(new Index1D(condnumstates));
                        MergeCountMap = accelerate.Allocate1D<double>(new Index1D(mergenumstates));

                        // print1d(CondNormBuffer.GetAsArray1D());
                        // print1d(MergeNormBuffer.GetAsArray1D());

                        BuildJointFreqKern(CondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, CondNormBuffer.View, JointBuffer1.View);
                        //EntropyBuffer1.GetAsArray1D();
                        //EntropyBuffer1.GetAsArray1D();
                        //ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);

                        BuildJointFreqKern(MergeNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, MergeNormBuffer.View, JointBuffer2.View);
                        //EntropyBuffer1.GetAsArray1D();

                        //EntropyBuffer1.GetAsArray1D();
                        //ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);

                        BuildFreqKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, CondCountMap.View);
                        //ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);
                        
                        //EntropyBuffer1.GetAsArray1D();
                        //print1d(MergeNormBuffer.GetAsArray1D());
                        BuildFreqKern(MergeNormBuffer.Extent.ToIntIndex(), MergeNormBuffer.View, MergeCountMap.View);
                        //EntropyBuffer1.GetAsArray1D();
                        //ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);


                        //EntropyBuffer1.GetAsArray1D();
                        Console.WriteLine("Cond and second");
                        print1d(CondNormBuffer.GetAsArray1D());
                        print1d(SecondNormBuffer.GetAsArray1D());
                        print1d(CountNonNaN.GetAsArray1D());
                        Console.WriteLine(targetvar.GetLength(0));

                        Console.WriteLine("Merge and second");
                        print1d(MergeNormBuffer.GetAsArray1D());
                        print1d(SecondNormBuffer.GetAsArray1D());
                        print1d(CountNonNaN.GetAsArray1D());
                        Console.WriteLine(targetvar.GetLength(0));
                        IndexedCalcConditionalEntropyKern(CondNormBuffer.Extent.ToIntIndex(), JointBuffer1.View, CondCountMap.View, SecondNormBuffer.View, CondNormBuffer.View,EntropyBuffer1.View, targetvar.GetLength(0));
                        IndexedCalcConditionalEntropyKern(MergeNormBuffer.Extent.ToIntIndex(), JointBuffer2.View, MergeCountMap.View, SecondNormBuffer.View, MergeNormBuffer.View,EntropyBuffer2.View, targetvar.GetLength(0));

                        print2d(JointBuffer1.GetAsArray2D());
                        print2d(JointBuffer2.GetAsArray2D());

                        //Console.Write("Cond Entropy");
                        ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);
                        print1d(targetvar);
                        print1d(arr[j]);
                        Console.Write("Cond Entropy");
                        

                        Console.WriteLine(ent1);
                        ent2 = EntropyBuffer2.GetAsArray1D()[0] / Math.Log(LOG_BASE);
                        print1d(targetvar);
                        print1d(arr[i]);
                        print1d(SecondNormBuffer.GetAsArray1D());
                        
                        print1d(FirstNormBuffer.GetAsArray1D());
                        print1d(CondNormBuffer.GetAsArray1D());
                        print1d(MergeNormBuffer.GetAsArray1D());
                        Console.Write("Merge Entropy");
                        Console.WriteLine(ent2);

                        output.Add(new OutputGroup(columnnames[i], columnnames[j], ent1-ent2));
                        //accelerate.DefaultStream.Synchronize();
                    }
                }
            }

            return output.ToArray();

        }
        public double calculateConditionalMutualInformation(double[] firstVector, double[] secondVector, double[] conditionVector){
            
            //FirstCondEntropy (secondvector, conditionvector)
            //SecondCondEntropy (secondvector, mergedvector)
            Stopwatch watch = new Stopwatch();
            watch.Start();
            Accelerator accelerate = this.dev.CreateAccelerator(this.context);
            var FirstBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            var CondBuffer = accelerate.Allocate1D<double>(new Index1D(conditionVector.GetLength(0)));
            //var testIndex = new LongIndex2D(10000, 1000000);
            var FirstNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            var SecondNormBuffer = accelerate.Allocate1D<double>(new Index1D(secondVector.GetLength(0)));
            var CondNormBuffer = accelerate.Allocate1D<double>(new Index1D(conditionVector.GetLength(0)));

            var mergedBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));
            //var AdjMergeBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));

            var MergeNormBuffer = accelerate.Allocate1D<double>(new Index1D(firstVector.GetLength(0)));

            FirstBuffer.CopyFromCPU(firstVector);
            SecondBuffer.CopyFromCPU(secondVector);
            CondBuffer.CopyFromCPU(conditionVector);

            var FirstMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var FirstMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var SecondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var SecondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var CondMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var CondMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var MergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var MergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var PreMergeMaxVal = accelerate.Allocate1D<int>(new Index1D(1));
            var PreMergeMinVal = accelerate.Allocate1D<int>(new Index1D(1));

            var EntropyBuffer1 = accelerate.Allocate1D<double>(new Index1D(1));
            var EntropyBuffer2 = accelerate.Allocate1D<double>(new Index1D(1));

            var StateCount = accelerate.Allocate1D<int>(new Index1D(1));

            var CountNonNaN = accelerate.Allocate1D<double>(new Index1D(1));



            var GetMaxValKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(TestGetMaxMinValKernal);
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
                ArrayView2D<int, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                int, int>(mergeArraysAdjKernel);
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

            var setBuffToValueKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setBuffToValueKernal);

            var BuildFreqKern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                BuildFreqAdjKernel);

        

            var RefactorPart1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                int>(
                refactorKernal);

            var refactorPart1Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                refactorPart1Phase1Kernal);

            var refactorPart1Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                refactorPart1Phase2Kernal);


            var refactorPart2Phase1Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
                >(
                refactorPart2Phase1Kernal);

            var refactorPart2Phase2Kern = accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>
                >(
                refactorPart2Phase2Kernal);

            var countnonNaNKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(countnonNaN);
            
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

            normalizeArrayKern(FirstBuffer.Extent.ToIntIndex(), FirstBuffer.View, FirstNormBuffer.View, FirstMinVal.View);
            normalizeArrayKern(SecondBuffer.Extent.ToIntIndex(), SecondBuffer.View, SecondNormBuffer.View, SecondMinVal.View);
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



            var StateMap = accelerate.Allocate2DDenseX<int>(new Index2D(firstnumstates + 1, secondnumstates+ 1)); // accelerate.Allocate1D<int>(new Index1D((firstnumstates + 1)*(secondnumstates+ 1)));

            //outputVector = mergedBuffer.GetAsArray1D();
            
            setBuffToValueKern(StateCount.Extent.ToIntIndex(), StateCount.View, 1);
            //print1d(FirstNormBuffer.GetAsArray1D());
            //print1d(SecondNormBuffer.GetAsArray1D());
            //Console.ReadLine();
            mergeArraysKern(mergedBuffer.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, StateMap.View, mergedBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);
           // mergeArraysKern(StateCount.Extent.ToIntIndex(), FirstNormBuffer.View, SecondNormBuffer.View, StateMap.View, mergedBuffer.View,StateCount.View, firstnumstates, FirstNormBuffer.Extent.ToIntIndex().X);

            //outputVector = mergedBuffer.GetAsArray1D();
            // print1d(outputVector);
            // Console.WriteLine("InHere");
            // print1d(SecondNormBuffer.GetAsArray1D());
            // print1d(FirstNormBuffer.GetAsArray1D());
            //print1d(mergedBuffer.GetAsArray1D());
            // Console.ReadLine();
            InitMaxMinKern(PreMergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, PreMergeMaxVal.View, PreMergeMinVal.View);
            GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, PreMergeMaxVal.View, PreMergeMinVal.View);
            //accelerate.DefaultStream.Synchronize();
            int premergenumstates = PreMergeMaxVal.GetAsArray1D()[0];
            var TempBuffer = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));
            var TempBuffer2 = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));

            //var Temp2Buffer = accelerate.Allocate1D<double>(new Index1D(premergenumstates +1));

            var HolderBuffer = accelerate.Allocate1D<double>(new Index1D(1));
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
            setBuffToValueDoubleKern(TempBuffer2.Extent.ToIntIndex(), TempBuffer2.View, 1.0);

            setBuffToValueDoubleKern(HolderBuffer.Extent.ToIntIndex(), HolderBuffer.View, 0.0);

            refactorPart2Phase1Kern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, TempBuffer.View, TempBuffer2.View, HolderBuffer.View);
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

            Console.WriteLine("Refactor Arrs Part2:");

            //Console.WriteLine(watch.ElapsedMilliseconds);
            watch.Reset();
            InitMaxMinKern(MergeMaxVal.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            GetMaxValKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeMaxVal.View, MergeMinVal.View);
            //print1d(MergeMaxVal.GetAsArray1D());

            // print1d(MergeMaxVal.GetAsArray1D());
            // print1d(MergeMinVal.GetAsArray1D());
            // print1d(mergedBuffer.GetAsArray1D());
            // Console.ReadLine();

            normalizeArrayKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, MergeNormBuffer.View, MergeMinVal.View);

            int mergenumstates = MergeMaxVal.GetAsArray1D()[0];

            setBuffToValueDoubleKern(CountNonNaN.Extent.ToIntIndex(), CountNonNaN.View, 1.0);
            countnonNaNKern(mergedBuffer.Extent.ToIntIndex(), mergedBuffer.View, CountNonNaN.View);


            Console.WriteLine("NumStates");
            Console.WriteLine(secondnumstates);

            Console.WriteLine(condnumstates);

            Console.WriteLine(mergenumstates);
            Console.WriteLine(MergeMinVal.GetAsArray1D()[0]);
            Console.WriteLine("PreMergeNumStates");
            Console.WriteLine(premergenumstates);

            Console.WriteLine(CountNonNaN.GetAsArray1D()[0]);


            //print1d(mergedBuffer.GetAsArray1D());
            //Console.ReadLine();

            var JointBuffer1 = accelerate.Allocate2DDenseX<double>(new Index2D(secondnumstates +1, condnumstates+1));
            var JointBuffer2 = accelerate.Allocate2DDenseX<double>(new Index2D(secondnumstates +1, mergenumstates+1));

            //EntropyBuffer1.GetAsArray1D();

            
            //var SecondCountMap = accelerate.Allocate1D<double>(new Index1D(secondnumstates));
            var CondCountMap = accelerate.Allocate1D<double>(new Index1D(condnumstates));
            var MergeCountMap = accelerate.Allocate1D<double>(new Index1D(mergenumstates));

            // print1d(CondNormBuffer.GetAsArray1D());
            // print1d(MergeNormBuffer.GetAsArray1D());

            BuildJointFreqKern(CondNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, CondNormBuffer.View, JointBuffer1.View);
            //EntropyBuffer1.GetAsArray1D();
            //EntropyBuffer1.GetAsArray1D();

            BuildJointFreqKern(MergeNormBuffer.Extent.ToIntIndex(), SecondNormBuffer.View, MergeNormBuffer.View, JointBuffer2.View);
            //EntropyBuffer1.GetAsArray1D();

            //EntropyBuffer1.GetAsArray1D();

            BuildFreqKern(CondNormBuffer.Extent.ToIntIndex(), CondNormBuffer.View, CondCountMap.View);
            //EntropyBuffer1.GetAsArray1D();
            //print1d(MergeNormBuffer.GetAsArray1D());
            BuildFreqKern(MergeNormBuffer.Extent.ToIntIndex(), MergeNormBuffer.View, MergeCountMap.View);
            //EntropyBuffer1.GetAsArray1D();


            //EntropyBuffer1.GetAsArray1D();
            IndexedCalcConditionalEntropyKern(CondNormBuffer.Extent.ToIntIndex(), JointBuffer1.View, CondCountMap.View, SecondNormBuffer.View, CondNormBuffer.View,EntropyBuffer1.View, secondVector.GetLength(0));
            IndexedCalcConditionalEntropyKern(MergeNormBuffer.Extent.ToIntIndex(), JointBuffer2.View, MergeCountMap.View, SecondNormBuffer.View, MergeNormBuffer.View,EntropyBuffer2.View, secondVector.GetLength(0));

            double ent1 = EntropyBuffer1.GetAsArray1D()[0] / Math.Log(LOG_BASE);


            double ent2 = EntropyBuffer2.GetAsArray1D()[0] / Math.Log(LOG_BASE);

            return ent1-ent2;


        }
        public double ConditiponalMIHolder(double[] firstVector, double[] secondVector, double[] conditionVector){
            
        


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
        static void replaceNaNKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView
            ){
            if(Double.IsNaN(inputView[index]) ){
                inputView[index] = -1.0;
            }
        }
        static void refactorKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            int length){
            int holder =0;

            for(int i = 0; i < length; i++ ){
                if(inputView[index] == inputView[new Index1D(i)] && index.X != i ){
                    holder = 1;   

                }
            }
            if(holder == 0){
                inputView[index] = Double.NaN;
            }
        }
        // static void fixFirst(Index1D index,
        //     ArrayView1D<double, Stride1D.Dense> inputView){
        //     if(Double.IsNaN(inputView[index]))
        // }
        static void refactorPart1Phase1Kernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView){
            if(!Double.IsNaN(inputView[index])){// && inputView[index] > 1000){
                Atomic.Add(ref holderView[new Index1D((int)inputView[index])], 1.0);

            }
        }
        static void refactorPart1Phase2Kernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView){
            if(holderView[new Index1D((int)inputView[index])] <= 1.0){
                inputView[index] = Double.NaN;
            }
        }
            
        static void refactorPart2Phase1Kernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView,
            ArrayView1D<double, Stride1D.Dense> holderView2,
            ArrayView1D<double, Stride1D.Dense> sharedmem){
            
                
            if(!Double.IsNaN(inputView[index]) && holderView[(int)inputView[index]] > 1 &&  (Atomic.Add(ref holderView2[(int)inputView[index]], 1.0) == 1)){

                // //double x = Atomic.Add(ref holderView[(int)inputView[index]], Atomic.Add(ref sharedmem[new Index1D(0)], 1.0));
                // if(holderView[(int)inputView[index]] > 1.0){
                //     //holderView[(int)inputView[index]] = x;
                //     //Atomic.Add(ref holderView[(int)inputView[index]], sharedmem[new Index1D(0)]);
                // }
                // else{
               
                holderView[(int)inputView[index]] =  Atomic.Add(ref sharedmem[new Index1D(0)], 1.0);
                    

                


            }
            // else if(!Double.IsNaN(inputView[index]) && holderView[(int)inputView[index]] <=1 ){
            //     holderView[(int)inputView[index]] =  Double.NaN;
            // }
            // else{
            //     holderView[(int)inputView[index]] = 6.9;
            // }

            
        }   
        static void countnonNaN(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> output){
            if(!Double.IsNaN(inputView[index])){
                Atomic.Add(ref output[new Index1D(0)], 1.0);
            }

        }
        static void refactorPart2Phase2Kernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> holderView){
            
                
            if(!Double.IsNaN(inputView[index])){
                inputView[index] = holderView[(int)inputView[index]];

            }
            else if(index.X == 0 && Double.IsNaN(inputView[index] )){
                inputView[new Index1D(0)] = 0.0;
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
        static void mergeArraysAdjKernel(Index1D index,
            ArrayView1D<double, Stride1D.Dense> FirstNormView,
            ArrayView1D<double, Stride1D.Dense> SecondNormView,
            ArrayView2D<int, Stride2D.DenseX> StateMap,
            ArrayView1D<double, Stride1D.Dense> OutputView,
            ArrayView1D<int, Stride1D.Dense> SCount,
            int firstnumstates,
            int length
            ){
            Index2D curindex;
            //int statecount = 1;
            //Probably doesnt work
            if(!Double.IsNaN(FirstNormView[index]) &&!Double.IsNaN(SecondNormView[index]) ){

            
                curindex =  new Index2D((int)FirstNormView[index], (int)SecondNormView[index]);// + ((int)SecondNormView[index] * firstnumstates);
                
                if(StateMap[curindex] == 0){
                    //Atomic.Add(ref SCount[new Index1D(0)], 1);
                    StateMap[curindex] = Atomic.Add(ref SCount[new Index1D(0)], 1);
                }
                OutputView[index] = (double)StateMap[curindex];
            }
                
            

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
            if(!(Double.IsNaN(FirstView[index])) && !(Double.IsNaN(SecondView[index]))){
                Index2D ind2 = new Index2D((int)FirstView[index], (int)SecondView[index]);

                if(FreqView[ind2] > 0 && condView[ind2.Y] > 0){
                    val = -1 * ((FreqView[ind2]/(double)length) * Math.Log( (FreqView[ind2]/(double)length) /  (condView[ind2.Y]/(double)length)));
                    val = val / FreqView[ind2];
                    Atomic.Add(ref entropy[new Index1D(0)], val);
                }
            }
            else{


                
                val = -1 * ((1/(double)length) * Math.Log( (1/(double)length) /  (1/(double)length)));
                    
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
            if(aView[index] != Double.NaN){
                Atomic.Max(ref MaxVal[new Index1D(0)],  (int)Math.Floor(aView[index]));
                Atomic.Min(ref MinVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                //Atomic.Add(ref MinVal[new Index1D(0)], 1);
            }
            
            
            

        }
        static void TestGetMaxMinValKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal)
        {
            if(!Double.IsNaN(aView[index])){
                Atomic.Max(ref MaxVal[new Index1D(0)],  (int)Math.Floor(aView[index]));
                Atomic.Min(ref MinVal[new Index1D(0)], (int)Math.Floor(aView[index]));
                //Atomic.Add(ref MinVal[new Index1D(0)], 1);
            }
            
            
            

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
        static void setBuffToValue2DIntKernal(Index2D index, 
            ArrayView2D<int, Stride2D.DenseX> buff, 
            int setvalue)
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
            if(!(Double.IsNaN(input[index]))){
                Atomic.Add(ref output[(int)Math.Floor(input[index])], 1.0);

            }
            //if(Math.Floor(input[index.X]) == Math.Floor(input[index.Y])){
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
            if(!(Double.IsNaN(first[index])) && !(Double.IsNaN(second[index]))){
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

        void testmulti(int length, int numvariables, int cap){
            //Console.WriteLine("In Test Mutli");
            Stopwatch stop = new Stopwatch();

            List<double[]> dataframe = new List<double[]>();
            Random rand = new Random();

            double[] temp;
            double[] mainvar = new double[length];
            string[] colnames = new string[numvariables];
            for(int j =0; j < length; j++){
                mainvar[j] = rand.NextDouble() * cap;
            }
            for(int i =0; i < numvariables; i++){
                colnames[i] = i.ToString();
                temp = new double[length];
                for(int j =0; j < length; j++){
                    temp[j] = rand.NextDouble() * cap;
                }
                dataframe.Add(temp);
            }
            Console.WriteLine("Here");
            stop.Start();
            OutputGroup[] best = findBestCondMutInf(mainvar, dataframe, colnames);
            stop.Stop();
            // Console.Write("[");
            // for(int i = 0; i < best.GetLength(0); i++){   
            //     Console.Write(best[i]);
            //     Console.Write(", ");
            // } 
            // Console.WriteLine("]");

            Console.Write("Total Time: ");
            Console.WriteLine(stop.ElapsedMilliseconds);

        }
        void test(int length){
            Stopwatch stop = new Stopwatch();
            Console.Write("LENGTH =");
            Console.WriteLine(length);
            Random rand = new Random();
            double[] a = new double[length];
            for (int i = 0; i < length; i++) {
              a[i] = rand.NextDouble() * 1000;
            }
            double[] b = new double[length];
            for (int i = 0; i < length; i++) {
              b[i] = rand.NextDouble() * 100;
            }
            double[] c = new double[length];
            for (int i = 0; i < length; i++) {
              c[i] = rand.NextDouble() * 100;
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
            Console.Write("Elapsed time f or Calculate Conditional Entropy: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            calculateJointEntropy(a,b);
            stop.Stop();
            Console.Write("Elapsed time for Calculate Joint Entropy: ");
            Console.WriteLine(stop.ElapsedMilliseconds);
            stop.Reset();

            stop.Start();
            calculateConditionalMutualInformation(a,b,c);
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
            double[] a = new []{4.2, 5.43, 3.221, 7.34235, 1.931, 1.2, 5.43, 8.0, 7.34235, 1.931};
            double[] b =  new []{2.2, 3.43, 1.221, 1.34235, 7.931, 7.2, 4.43, 7.0, 7.34235, 34.931};
            double[] d =  new []{2.2, 6.43, 2.221, 2.34235, 3.931, 3.2, 7.43, 7.0, 9.34235, 12.931};
            string[] s = new [] {"test", "b", "c"};
            //double[] b = new[] {2.2, 3.43, 1.221, 9.34235, 7.931, 12.2, 4.43, 13.0, 14.34235, 34.931};
            List<double[]> dataframe = new List<double[]>();
            //dataframe.Add(a);
            dataframe.Add(b);
            dataframe.Add(d);

            double[] temp = new double [1000];
            MI m = new MI();
            //Console.WriteLine(m.calculateConditionalMutualInformation(a,b,d));
            //Console.WriteLine(m.calculateConditionalMutualInformation(b,a,d));


            OutputGroup[] best = m.findBestCondMutInf(a, dataframe, s);
            for(int i = 0; i < best.GetLength(0); i++){
                Console.WriteLine(best[i]);
            }
            Console.ReadLine();
            Random rd = new Random();
            List<double[]> newlist = new List<double[]>();
            // for(int i = 0; i < 100; i ++){
            //     temp = new double [1000];
            //     for(int j = 0; j < 1000; j++){
            //         temp[j] = rd.NextDouble() * 100;
            //     }
            //     newlist.Add(temp);
            // }
            // Console.WriteLine(newlist[0][0]);
            // Console.WriteLine(newlist[1][0]);
            
            //m.testmulti(1000,100,10);
            //Console.WriteLine("Refactoring");
            //m.refactorArray(a);
            //Console.ReadLine();
      //       Console.WriteLine("In Multi");
	    	// List<double> entropyarr = m.MulticalculateMutualInformation(newlist);
      //       Console.WriteLine("Done");
      //       m.print1d(entropyarr.ToArray());
      //       Console.ReadLine();
            double[] c = new double [10];
	    	Context context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
            
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

            for (int i = 2; i < 7; i++){
                for(int j = 1; j < 7; i++){
                    for(int k = 1; k< 5; k++){
                        Console.Write("Length: ");
                        Console.Write(Math.Pow(10,i));
                        Console.Write(" | Numvariables: ");
                        Console.Write(Math.Pow(10,j));
                        Console.Write("| Cap: ");
                        Console.WriteLine(Math.Pow(10,k));
                        m.testmulti((int)Math.Pow(10,i),(int)Math.Pow(10,j),(int)Math.Pow(10,k));
                        Console.WriteLine();
                    }
                }
              //m.test((int)Math.Pow(10,i));
            }
            Console.WriteLine("calc cond mutInf");
            Console.WriteLine(m.calculateConditionalMutualInformation(a,b, d));
            // Console.ReadLine();
            // m.print1d(a);
            // m.print1d(m.refactorToMinimizeSize(a));
            // //Console.ReadLine();
            // m.print1d(b);
            // m.mergeArrays(a,b, ref c);
            // m.print1d(c);

            // // InitMaxMinKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);
            // // Console.WriteLine(MaxVal.GetAsArray1D()[0]);
            // // Console.WriteLine(MinVal.GetAsArray1D()[0]);
            // // GetMaxValKern(MVBuffer.Extent.ToIntIndex(), MVBuffer.View, MaxVal.View, MinVal.View);

            // // Console.WriteLine(MaxVal.GetAsArray1D()[0]);
            // // Console.WriteLine(MinVal.GetAsArray1D()[0]);
            // // BuildFreqKern(new Index2D(MVBuffer.Extent.ToIntIndex().X,MVBuffer.Extent.ToIntIndex().X), MVBuffer.View, FreqBuffer.View);
            // // m.print1d(MVBuffer.GetAsArray1D());
            // // m.print1d(FreqBuffer.GetAsArray1D());

            // // CalcEntropyKern(FreqBuffer.Extent.ToIntIndex(), FreqBuffer.View, EntropyBuffer.View, 10);
            // // m.print1d(EntropyBuffer.GetAsArray1D());
            // // Console.WriteLine(m.calculateMutualInformation(a,b));
            // // Console.WriteLine("MI ^^^");

            // Console.WriteLine(m.calculateConditionalEntropy(a,m.refactorToMinimizeSize(b)));
            // Console.WriteLine("Conditional Entropy ^^^");
            // // /Console.ReadLine();
            // Console.WriteLine(m.calculateJointEntropy(a,b));
            // Console.WriteLine("JOINT ENTROPY^^^");
            // Console.WriteLine(m.calculateEntropy(b));

            // m.mergeArrays(a,b,ref c);
            // Console.WriteLine(m.calculateEntropy(c));
            // //m.print1d(c);
	    	Console.WriteLine("Hello World");
	    }
	        
	}
}		  