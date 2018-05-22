using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class LSTM_TrivialExampl
    {
        private System.DateTime PROCESS_START;
        public LSTM_TrivialExampl()
        {
            PROCESS_START = DateTime.Now;
            Console.WriteLine("LSTM Example");
        }
        public void Example()
        {
            Random rndm = new Random(0);
            int nLayers = 1;
            int nIn = 1;
            int nOut = nIn;
            int memCellCount = 11;
            int nT = 11;
            double LR = 0.2;

            LSTMRNN NET = new LSTMRNN(nIn, memCellCount, nOut, nLayers, nT, rndm,0.6);
            double[] sequence = new double[]
            { 
                0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1
            };
            int maxepoch = 20000;
            int iter = 0;
            double[] output = new double[nOut];
            while (iter < maxepoch)
            {
                for (int W = 0; W < sequence.Length-2; ++W)
                {

                    
                    Array.Copy(NET.ComputeOutputs(new double[] { sequence[W] }), output, nOut);
                    NET.ComputeInputError(Measure.TminusO(new double[] { sequence[W+1] }, output), LR, rndm, 1, 10, 5);
                    Console.Write(output[0].ToString() + "   "); // write values

                    if (NET.DROPOUT_RATE > 0)
                        if (iter % 500 == 0)
                        {
                            NET.adjust_DROPOUT(NET.DROPOUT_RATE - 0.01);
                            if(LR>0.003)
                                LR -= 0.002;
                        }

                }
                READOUT(iter, nOut);
                ++iter;
            }
        }
        private void READOUT(int T, int nOut)
        {
            //set color!
            ConsoleColor prvTxtClr = Console.ForegroundColor;
            ConsoleColor prvBackgrndClr = Console.BackgroundColor;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.Yellow;
            //Write values!
            Console.WriteLine("\nITERATION: {0} \nTIME: [START: {1}], [NOW: {2}]", T, PROCESS_START, DateTime.Now);
            //set color back!
            Console.ForegroundColor = prvTxtClr;
            Console.BackgroundColor = prvBackgrndClr;
        }
    }
}

