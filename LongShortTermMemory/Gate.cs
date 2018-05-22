using System;
namespace LongShortTermMemory
{
    class Gate
    {
        int NumberInput;
        int NumberNodes;
        bool Logistic;
        /// <summary>
        /// store the pre-activation-output for BPTT
        /// </summary>
        double[][] PreActivOutput;

        double[][] W;
        public double[][] RW;
        double[] B;
        public double[][] PostActvOut;
        public double[][] eSgnl;
        double[][] W_Grads;
        double[][] RW_Grads;
        double[] B_Grads;
        public double DROPOUT_RATE;
        /// <summary>
        /// indices of zeros and ones that determine how much dropout is being done
        /// </summary>
        double[] DropoutIndices;
        

        /// <summary>
        /// the dropout technique randomly selects a percentage of nodes (usually 20%)  to register as output 0, and does update gradients for that cycle
        /// </summary>
        /// <param name="nInput"></param>
        /// <param name="nNodes"></param>
        /// <param name="nTimesteps"></param>
        /// <param name="rnd"></param>
        /// <param name="logistic"></param>
        /// <param name="dropout"></param>
        public Gate(int nInput, int nNodes, int nTimesteps, Random rnd, bool logistic = false, double dropoutRate = 0.2)
        {
            NumberInput = nInput;
            NumberNodes = nNodes;
            Logistic = logistic;
            PreActivOutput = M.MakeMatrix(nTimesteps, nNodes);
            PostActvOut = M.MakeMatrix(nTimesteps, nNodes);
            eSgnl = M.MakeMatrix(nTimesteps, nNodes); // changed from nin to nNodes 1.19.17
            W = new double[nNodes][];
            RW = new double[nNodes][];
            B = M.InitializeWeights(nNodes, rnd);
            for (int N = 0; N < nNodes; ++N)
            {
                W[N] = M.InitializeWeights(nInput, rnd);
                RW[N] = M.InitializeWeights(nNodes, rnd);
            }
            W_Grads = M.MakeMatrix(nNodes, nInput);
            RW_Grads = M.MakeMatrix(nNodes, nNodes);
            B_Grads = new double[nNodes];
            if (dropoutRate < 1.0) //greater than 1 would be greater than 100% lol
                DROPOUT_RATE = dropoutRate;
            else
                DROPOUT_RATE = 0.0;
            DropoutIndices = M.Random_Dropout_Indices(nNodes, dropoutRate, rnd); //set the first epochs dropout indices

        }
        public double[] ComputeOutputs(double[] input, double[] prevBlockOutput, int t)
        {
            return PostActvOut[t] = NonLin.Activ(PreActivOutput[t] = M.ElMult(DropoutIndices, M.Sum(new double[][] {
                M.Dot(W, input), M.Dot(RW, prevBlockOutput), B })), Logistic);
        }
        public double[] ComputeInputError(double[] Error,double[] input_t, double[] prevBlockOutput, int t)
        {//does grads too
            eSgnl[t] = M.ElMult(Error, NonLin.Activ(PreActivOutput[t],Logistic,true)); // gate error backpropped through nonlin
            W_Grads = M.Sum(W_Grads, M.Outer(eSgnl[t], input_t)); 
            RW_Grads = M.Sum(RW_Grads, M.Outer(eSgnl[t], prevBlockOutput)); // recurrent gradients
            B_Grads = M.Sum(B_Grads, eSgnl[t]);
            return M.Dot(M.T(W), eSgnl[t]);
        }
        public void UpdateWeights(Random rnd, double LearnRate = 0.00001, int maxGrad = 1, int clipToDecimal = 15)
        {//yes having these operations each in their own forloop is way inneficient, but is easier to read for code example
            W = M.Sum(W, M.ElMult(DropoutIndices, M.Clip(W_Grads, maxGrad, clipToDecimal, LearnRate)));
            W_Grads = M.MakeMatrix(NumberNodes, NumberInput);
            RW = M.Sum(RW, M.ElMult(DropoutIndices,M.Clip(RW_Grads, maxGrad, clipToDecimal, LearnRate)));
            RW_Grads = M.MakeMatrix(NumberNodes, NumberNodes);
            B = M.Sum(B, M.ElMult(DropoutIndices, M.Clip(B_Grads, maxGrad, clipToDecimal, LearnRate)));
            B_Grads = new double[NumberNodes];
            DropoutIndices = M.Random_Dropout_Indices(DropoutIndices.Length, DROPOUT_RATE, rnd); //set the next epochs dropout indices
        }
    }
}
