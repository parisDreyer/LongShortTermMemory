using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class LSTMRNN
    {
        int NumberInput;
        int MemoryCellCount;
        int NumberOutput;
        int NumberLayers;
        public double DROPOUT_RATE;
        public double[] MostRecentInput;
        public double[] MostRecentOutput;
        public double[] InputErrorSignal;
        Block[] Layers;
        public LSTMRNN(int nIn, int nMemCells, int nOut, int nLayers, int nT, Random rnd, double dropoutRate=0.2)
        {
            NumberInput = nIn;
            MemoryCellCount = nMemCells;
            NumberOutput = nOut;
            NumberLayers = nLayers;
            DROPOUT_RATE = dropoutRate;
            MostRecentInput = new double[nIn];
            MostRecentOutput = new double[nOut];
            InputErrorSignal = new double[nIn];
            Layers = new Block[nLayers];
            for (int N = 0; N < nLayers; ++N)
                if (N == 0 && nLayers > 1) Layers[N] = new Block(nIn, nMemCells, nMemCells, nT, rnd, DROPOUT_RATE);
                else if (N < nLayers - 1) Layers[N] = new Block(nMemCells, nMemCells, nMemCells, nT, rnd, DROPOUT_RATE);
                else if(N==0 && nLayers==1) Layers[N] = new Block(nIn, nMemCells, nOut, nT, rnd, DROPOUT_RATE); //output layer if only one layer
                else Layers[N] = new Block(nMemCells, nMemCells, nOut, nT, rnd, DROPOUT_RATE); //output layer
        }
        public double[] ComputeOutputs(double[] input)
        {
            Array.Copy(input, MostRecentInput, NumberInput);
            MostRecentOutput = (double[])input.Clone(); // zero out previous outputs 
            for (int L = 0; L < NumberLayers; ++L) // sum next outputs from each block
                MostRecentOutput = Layers[L].ComputeOutputs(MostRecentOutput);
            return MostRecentOutput;
        }
        public double[] ComputeInputError(double[] Delta, double LR, Random rnd, int maxGrad = 1, int lastDec = 15, int trunc = 5)
        {
            InputErrorSignal = (double[])Delta.Clone();
            for (int L = NumberLayers-1; L>=0; --L)
                InputErrorSignal = (double[])Layers[L].ComputeInputError(InputErrorSignal,rnd, LR, maxGrad, lastDec, trunc).Clone();
            return InputErrorSignal;
        }
        public void adjust_DROPOUT(double newDropoutRate) {
            DROPOUT_RATE = newDropoutRate;
            for (int L = 0; L < NumberLayers; ++L)
                Layers[L].adjust_DROPOUT(DROPOUT_RATE);
        }
    }
}
