using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class RNNLayer
    {
        public int NumberInput;
        public int NumberNodes;
        public int NumberTimeSteps;
        /// <summary>
        /// Current TimeStep
        /// </summary>
        public int T;
        /// <summary>
        /// default is tanh, but if logistic, then use logistic sigmoid instead
        /// </summary>
        bool Logistic;
        
        public double[][] Input;
        public double[][] Output;
        /// <summary>
        /// A 2D matrix with rows=NumberInput and columns=NumberNodes
        /// </summary>
        public double[][] W;
        public double[][] RW;
        public double[] B;

        public double[][] W_Grads;
        public double[][] RW_Grads;
        public double[] B_Grads;
        /// <summary>
        /// Need at least 2 timesteps for the network to be recurrent
        /// </summary>
        /// <param name="nInput"></param>
        /// <param name="nNodes"></param>
        /// <param name="nTimeSteps"></param>
        /// <param name="rnd"></param>
        /// <param name="logistic"></param>
        public RNNLayer(int nInput, int nNodes, int nTimeSteps, Random rnd, bool logistic = false)
        {
            NumberInput = nInput;
            NumberNodes = nNodes;
            NumberTimeSteps = nTimeSteps;
            T = 0;
            Logistic = logistic;  

            Input = M.MakeMatrix(nTimeSteps, nInput);
            Output = M.MakeMatrix(nTimeSteps, nNodes);

            W = new double[nNodes][];
            RW = new double[nNodes][];
            B = M.InitializeWeights(nNodes, rnd);
            for (int W = 0; W < nNodes; ++W)
            {
                this.W[W] = M.InitializeWeights(nInput, rnd);
                RW[W] = M.InitializeWeights(nNodes, rnd);
            }
            
            
            W_Grads = M.MakeMatrix(nNodes, nInput);
            RW_Grads = M.MakeMatrix(nNodes, nNodes);
            B_Grads = new double[nNodes];
        }
        public double[] ComputeOutputs(double[] inputs)
        {
            Input.SetValue(inputs, T);
            double[] prev = new double[NumberNodes]; if (T > 0) prev = Output[T - 1];
            return Output[T] = NonLin.Activ(M.Sum(new double[][] { M.Dot(W,inputs), B, M.Dot(RW, prev)}), Logistic);
        }
        public double[] ComputeInputError(double[] outputdelta, int backsteps = 5, double LearnRate=0.00001)
        { 
            double[] E = M.ElMult(outputdelta, NonLin.Activ(Output[T], Logistic, true)); // current ErrorSignal
            BPTT_Grads(E, backsteps, LearnRate); // Gradients
            Clock();
            return M.Dot(M.T(W),E); // This Layer's Input Error (the Input's effect on the Weights' error)
        }

        /// <summary>
        /// backprops through time from currenttimestep 'T' to T-[truncated BBPT 'backsteps' idx], to get gradients for this layer's weights
        /// Returns Recurrent PreviousErrorSignal effect on current output error
        /// </summary>
        /// <param name="backsteps"></param>
        private void BPTT_Grads(double[] E, int backsteps = 5, double LearnRate = 0.00001, int maxGrad = 1, int clipToDecimal = 15)
        {
            if(T==0) UpdateWeights(LearnRate, maxGrad, clipToDecimal);
            W_Grads = M.Sum(W_Grads, M.Outer(E, Input[T]));
            B_Grads = M.Sum(B_Grads, E);
            for (int t = T; t >= T - backsteps + 1 && t >= (0+1); --t)//the +1 because we are backpropping just until 0
            {
                RW_Grads = M.Sum(RW_Grads, M.Outer(E, Output[t - 1]));
                E = M.ElMult(M.Dot(M.T(RW),E), NonLin.Activ(Output[t - 1], Logistic, true));
            }
        }
        private void UpdateWeights(double LearnRate=0.00001, int maxGrad = 2, int clipToDecimal = 15)
        {//yes having these operations each in their own forloop is way inneficient, but is easier to read for code example
            W = M.Sum(W, M.Clip(W_Grads, maxGrad, clipToDecimal, LearnRate));
            W_Grads = M.MakeMatrix(NumberNodes, NumberInput);
            RW = M.Sum(RW, M.Clip(RW_Grads, maxGrad, clipToDecimal, LearnRate));
            RW_Grads = M.MakeMatrix(NumberNodes, NumberNodes);
            B = M.Sum(B, M.Clip(B_Grads, maxGrad, clipToDecimal, LearnRate));
            B_Grads = new double[NumberNodes];
        }
        public void Clock()
        {
            ++T;
            if (T >= NumberTimeSteps)
                T = 0;
        }
    }
}
