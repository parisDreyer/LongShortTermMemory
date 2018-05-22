using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class LinearRegressionLayer
    {
        int NumberInput;
        int NumberOutput;
        double[][] W;
        double[] B;
        double[] B_Grads;
        double[][] W_Grads;
        double[] InputErrorSignal;
        double[] MostRecentInput;
        double[] MostRecentOutput;
        public double DROPOUT_RATE;
        /// <summary>
        /// indices of zeros and ones that determine how much dropout is being done
        /// </summary>
        double[] DropoutIndices;
        public LinearRegressionLayer(int nIn, int nOut, Random rnd, double dropoutRate = 0.2)
        {
            NumberInput = nIn;
            NumberOutput = nOut;
            MostRecentInput = new double[nIn];
            MostRecentOutput = new double[nOut];
            W = new double[nOut][];
            B = M.InitializeWeights(nOut, rnd);
            for (int N = 0; N < nOut; ++N)
            {
                W[N] = M.InitializeWeights(nIn, rnd);
            }
            InputErrorSignal = new double[nIn];
            W_Grads = M.MakeMatrix(nOut, nIn);
            B_Grads = new double[nOut];
            if (dropoutRate < 1.0) //greater than 1 would be greater than 100% lol
                DROPOUT_RATE = dropoutRate;
            else
                DROPOUT_RATE = 0.0;
            DropoutIndices = M.Random_Dropout_Indices(nOut, dropoutRate, rnd); //set the first epochs dropout indices
        }
        public double[] ComputeOutputs(double[] input)
        {
            Array.Copy(input, MostRecentInput, NumberInput);
            return MostRecentOutput = M.ElMult(DropoutIndices, M.Sum(M.Dot(W,input), B));
        }
        public double[] ComputeErrorSignal(double[] Delta)
        {
            W_Grads = M.Sum(W_Grads, M.Outer(Delta, MostRecentInput)); // gradients
            B_Grads = M.Sum(B_Grads, Delta);
            return InputErrorSignal = M.Dot(M.T(W),Delta); // error signal backpropped through weights
        }
        public void UpdateWeights(Random rnd, double lr = 0.00001, int maxGrd = 1, int clipToDecimal = 15)
        {
            W = M.Sum(W, M.ElMult(DropoutIndices, M.Clip(W_Grads, maxGrd, clipToDecimal, lr)));
            W_Grads = M.MakeMatrix(NumberOutput, NumberInput);
            B = M.Sum(B, M.ElMult(DropoutIndices, M.Clip(B_Grads, maxGrd, clipToDecimal, lr)));
            B_Grads = new double[NumberOutput];
            DropoutIndices = M.Random_Dropout_Indices(DropoutIndices.Length, DROPOUT_RATE, rnd); //set the next epochs dropout indices
        }
    }
}
