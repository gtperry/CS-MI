using System;

using ILGPU;
using ILGPU.Runtime;

namespace CSMI
{
    public class CommonKernels
    {
        public static void GetMaxMinValKernal(
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
            }
        }

        protected static void InitMaxMinKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<int, Stride1D.Dense> MaxVal,
            ArrayView1D<int, Stride1D.Dense> MinVal
        )
        {
            MaxVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
            MinVal[index] = (int)Math.Floor(aView[new Index1D(0)]);
        }

        protected static void normalizeArrayKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> inputView,
            ArrayView1D<double, Stride1D.Dense> outputView,
            ArrayView1D<int, Stride1D.Dense> minVal
        )
        {
            outputView[index] = Math.Floor(inputView[index]) - (double)minVal[new Index1D(0)];
        }

        protected static void BuildJointFreqKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView2D<double, Stride2D.DenseX> output
        )
        {
            if (!(Double.IsNaN(first[index])) && !(Double.IsNaN(second[index])))
            {
                Atomic.Add(ref output[new Index2D((int)first[index], (int)second[index])], 1.0);
            }
        }

        protected static void BuildJointFreqKernel2(
            Index1D index,
            ArrayView<double> first,
            ArrayView<double> second,
            ArrayView<double> output,
            long minSzudzikPair
        )
        {
            if (!(Double.IsNaN(first[index])) && !(Double.IsNaN(second[index])))
            {
                Atomic.Add(
                    ref output[
                        (Utils.szudzikPair((int)first[index] - minSzudzikPair, (int)second[index] - minSzudzikPair))
                    ],
                    1.0
                );
            }
        }

        protected static void BuildJointFreqKernel3(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> first,
            ArrayView1D<double, Stride1D.Dense> second,
            ArrayView2D<double, Stride2D.DenseX> output
        )
        {
            if (!(Double.IsNaN(first[index])) && !(Double.IsNaN(second[index])))
            {
                Atomic.Add(ref output[new Index2D((int)first[index], (int)second[index])], 1.0);
            }
        }

        ///<summary>Sets every element in <c>buff</c> to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        protected static void setBuffToValue2DKernal(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> buff,
            double setvalue
        )
        {
            buff[index] = setvalue;
        }

        ///<summary>Sets every element in <c>buff</c> to setvalue</summary>
        ///<param name="buff">buff</param>
        ///<param name="setvalue">setvalue</param>
        protected static void setBuffToValueDoubleKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> buff,
            double setvalue
        )
        {
            buff[index] = setvalue;
        }

        protected static void BuildFreqAdjKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            if (!(Double.IsNaN(input[index])))
            {
                Atomic.Add(ref output[(int)Math.Floor(input[index])], 1.0);
            }
        }

        protected static void BuildFreqAdjustedKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> input,
            ArrayView1D<double, Stride1D.Dense> output
        )
        {
            Atomic.Add(ref output[new Index1D((int)(input[index]))], 1.0);
        }

        protected static void TestGetMaxMinValKernal(
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
            }
        }
    }
}
