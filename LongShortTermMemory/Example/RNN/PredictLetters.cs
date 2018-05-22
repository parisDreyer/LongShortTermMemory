using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory.Example.RNN
{
    class PredictLetters
    {
        public PredictLetters() { Console.WriteLine("Running PredictLetters "); }
        public void Example()
        {
            Random rndm = new Random(0);
            int nLayers = 1;
            int nIn = 27;
            int[] nNodes = new int[] { 27 };
            int[] tSteps = new int[] { 2 };
            double LR = 0.001;

            RNNLayer[] Network = new RNNLayer[nLayers];
            for (int N = 0; N < nLayers; ++N)
                if(N == 0) Network[N] = new RNNLayer(nIn, nNodes[0], tSteps[0], rndm);
                else if(N<nLayers-1) Network[N] = new RNNLayer(nNodes[N-1], nNodes[N], tSteps[N], rndm);
                else Network[N] = new RNNLayer(nNodes[N - 1], nNodes[N], tSteps[N], rndm, true); // logsig b4 softmax

            string[] vocab = new string[] {" squid flax", " zephyr tow",};
            int maxepoch = 60000;
            int iter = 0;

            double[][][][] TrainingData = Alphabet.InitializeTrainingData(vocab);
            while (iter < maxepoch)
            {
                double[] output = new double[0];
                int[] S = Measure.Shuffle(TrainingData[0].Length, rndm);
                for (int W = 0; W < TrainingData[0].Length; ++W)
                {
                    for (int L = 0; L < TrainingData[0][S[W]].Length; ++L)
                    {
                        for(int N = 0; N<nLayers;++N) // FRWRD
                            if(N == 0) output = Network[N].ComputeOutputs(TrainingData[0][S[W]][L]);
                            else output = Network[N].ComputeOutputs(output);
                        output = NonLin.Softmax(output);
                        double[] E = new double[0]; // BPTT
                            for (int N = nLayers-1; N>=0; --N)
                                if(N == nLayers - 1) E = Network[N].ComputeInputError(Measure.TminusO(TrainingData[1][S[W]][L], output),tSteps[N],LR);
                                else E = Network[N].ComputeInputError(E, tSteps[N], LR);
                        Console.Write(Alphabet.Letters[Measure.MaxIndex(output)[0]]); // write values
                    }
                }
                ++iter;
            }
        }
    }
}
