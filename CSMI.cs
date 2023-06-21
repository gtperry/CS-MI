using System;

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

    public class CSMI
    {
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

            using TestSuite csmi_test = new TestSuite();

            if (option_selected == 1)
            {
                for (int i = 1; i < 11; i++)
                {
                    try
                    {
                        Console.WriteLine("Iteration: " + i);
                        csmi_test.testRandom((int)Math.Pow(10, i));
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
                csmi_test.runAllTests();
            }
        }
    }
}
