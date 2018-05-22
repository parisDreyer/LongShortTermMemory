using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class LSTM_LAYER
    {
        int NumberInput;
        int MemoryCellCount;
        int NumberOutput;
        public double[] MostRecentInput;
        public double[] MostRecentOutput;
        public double[] InputErrorSignal;
        Block[] Blocks;
        public LSTM_LAYER(int nBlocks, int nMemCells,int nOut, int nT, Random rnd)
        {
            NumberInput = nBlocks;
            MemoryCellCount = nMemCells;
            NumberOutput = nOut;
            MostRecentInput = new double[nBlocks];
            MostRecentOutput = new double[nOut];
            InputErrorSignal = new double[nBlocks];
            Blocks = new Block[nBlocks];
            for (int B = 0; B < nBlocks; ++B) Blocks[B] = new Block(1, nMemCells, nOut, nT, rnd);
        }
        public double[] ComputeOutputs(double[] input)
        {
            Array.Copy(input, MostRecentInput, NumberInput);
            MostRecentOutput = new double[NumberOutput]; // zero out previous outputs 
            for (int B = 0; B < NumberInput; ++B) // sum next outputs from each block
                MostRecentOutput = M.Sum(MostRecentOutput, Blocks[B].ComputeOutputs(new double[] { input[B] }));
            return MostRecentOutput;
        }
        public double[] ComputeInputError(double[] Delta, double LR,Random rnd, int maxGrad = 1,int lastDec = 15, int trunc = 5)
        {
            for (int B = 0; B < NumberInput; ++B)
                InputErrorSignal[B] = Blocks[B].ComputeInputError(Delta,rnd,LR,maxGrad,lastDec,trunc)[0];
            return InputErrorSignal;
        }
    }
}
