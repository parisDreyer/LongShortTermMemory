using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class Block
    {
        int NumberInputs;
        int NumberNodes;
        int NumberOutput;
        int NumberTimeSteps;
        int T;
        double DROPOUT_RATE;
        Gate Ig;
        Gate Zg;
        Gate Fg;
        Gate Og;

        double[][] MemoryCell;
        public double[][] BlockOutputs;
        public double[][] inputs;
        LinearRegressionLayer LRgrs;

        public Block(int nIn, int nNodes, int nOut, int nTimeSteps, Random rnd, double dropoutRate=0.2)
        {
            NumberInputs = nIn;
            NumberNodes = nNodes;
            NumberOutput = nOut;
            NumberTimeSteps = nTimeSteps;
            T = 0;
            DROPOUT_RATE = dropoutRate;
            inputs = M.MakeMatrix(nTimeSteps, nIn);
            BlockOutputs = M.MakeMatrix(nTimeSteps, nNodes);
            MemoryCell = M.MakeMatrix(nTimeSteps, nNodes);
            Ig = new Gate(nIn, nNodes, nTimeSteps, rnd, true, DROPOUT_RATE);
            Zg = new Gate(nIn, nNodes, nTimeSteps, rnd, false, DROPOUT_RATE);
            Fg = new Gate(nIn, nNodes, nTimeSteps, rnd, true, DROPOUT_RATE);
            Og = new Gate(nIn, nNodes, nTimeSteps, rnd, true, DROPOUT_RATE);
            LRgrs = new LinearRegressionLayer(nNodes, nOut, rnd, DROPOUT_RATE);
        }
        public double[] ComputeOutputs(double[] input)
        {
            inputs[T] = input;
            double[] Bprv = new double[NumberNodes]; // prev Block output
            double[] Sprv = new double[NumberNodes]; // prev memoryCell State
            if (T > 0) { Bprv = BlockOutputs[T - 1]; Sprv = MemoryCell[T - 1]; }
            MemoryCell[T] = M.Sum(M.ElMult(Zg.ComputeOutputs(input, Bprv, T), Ig.ComputeOutputs(input, Bprv, T)),
                M.ElMult(Fg.ComputeOutputs(input, Bprv, T), Sprv));
            return LRgrs.ComputeOutputs(BlockOutputs[T] = M.ElMult(NonLin.Activ(MemoryCell[T]), Og.ComputeOutputs(input, Bprv, T)));
        }
        /// <summary>
        /// trunc means 'truncated BPTT'
        /// </summary>
        /// <param name="E"></param>
        /// <param name="lr"></param>
        /// <param name="maxGrad"></param>
        /// <param name="lastDec"></param>
        /// <param name="trunc"></param>
        /// <returns></returns>
        public double[] ComputeInputError(double[] E, Random rnd, double lr = 0.00001, int maxGrad = 1, int lastDec = 15, int trunc=5)
        {
            double[] memE = new double[NumberNodes]; // next timestep memory cell error
            double[] RV = new double[NumberInputs];
            RV = BlockGateInptEr(memE = M.ElMult(E = LRgrs.ComputeErrorSignal(E), Og.PostActvOut[T], NonLin.Activ(MemoryCell[T], false, true)),T);
            if(T==NumberTimeSteps-1)
            for (int t = T; t >= T - trunc && t >= 1; --t) // BPTT
                if (t < T)
                {
                    E = M.Sum(new double[][] { E, M.Dot(M.T(Ig.RW),Ig.eSgnl[t+1]), M.Dot(M.T(Zg.RW), Ig.eSgnl[t+1]), M.Dot(M.T(Fg.RW), Fg.eSgnl[t+1]),
                        M.Dot(M.T(Og.RW), Og.eSgnl[t+1])});
                    memE = M.Sum(M.ElMult(E, Og.PostActvOut[t], NonLin.Activ(MemoryCell[t], false, true)), M.ElMult(memE, Fg.PostActvOut[t + 1]));
                    GateInptEr(memE, t);
                }
            Clock(); 
            if(T == 0)
            {
                UpdateWs(lr, rnd,maxGrad, lastDec); 
                MemoryCell = M.MakeMatrix(NumberTimeSteps,NumberNodes); 
            } // reset cell state and time steps
            return RV;
        }
        private double[] BlockGateInptEr(double[] memE, int t)
        { 
            double[] prvmc = new double[NumberNodes]; if(t>0) prvmc = MemoryCell[t - 1];
            return M.Sum(new double[][] {
            Og.ComputeInputError(M.ElMult(memE, NonLin.Activ(MemoryCell[t])), inputs[t], BlockOutputs[t], t),
            Fg.ComputeInputError(M.ElMult(memE, prvmc), inputs[t], BlockOutputs[t], t),
            Ig.ComputeInputError(M.ElMult(memE, Zg.PostActvOut[t]), inputs[t], BlockOutputs[t], t),
            Zg.ComputeInputError(M.ElMult(memE, Ig.PostActvOut[t]), inputs[t], BlockOutputs[t], t)});
        }
        private void GateInptEr(double[] memE, int t)
        {
            Og.ComputeInputError(M.ElMult(memE, NonLin.Activ(MemoryCell[t])), inputs[t], BlockOutputs[t], t);
            Fg.ComputeInputError(M.ElMult(memE, MemoryCell[t - 1]), inputs[t], BlockOutputs[t], t);
            Ig.ComputeInputError(M.ElMult(memE, Zg.PostActvOut[t]), inputs[t], BlockOutputs[t], t);
            Zg.ComputeInputError(M.ElMult(memE, Ig.PostActvOut[t]), inputs[t], BlockOutputs[t], t);
        }
        public void UpdateWs(double LR,Random rnd, int maxGrad = 1, int lastDec =15)
        {
            Og.UpdateWeights(rnd,LR,maxGrad,lastDec);
            Fg.UpdateWeights(rnd, LR, maxGrad, lastDec);
            Ig.UpdateWeights(rnd, LR, maxGrad, lastDec);
            Zg.UpdateWeights(rnd, LR, maxGrad, lastDec);
            LRgrs.UpdateWeights(rnd, LR, maxGrad, lastDec);
        }
        public void Clock()
        {
            ++T;
            if (T >= NumberTimeSteps)
                T = 0;
        }
        public void adjust_DROPOUT(double newDropoutRate) {
            DROPOUT_RATE = newDropoutRate;
            Og.DROPOUT_RATE = DROPOUT_RATE;
            Fg.DROPOUT_RATE = DROPOUT_RATE;
            Ig.DROPOUT_RATE = DROPOUT_RATE;
            Zg.DROPOUT_RATE = DROPOUT_RATE;
            LRgrs.DROPOUT_RATE = DROPOUT_RATE;
        }
    }
}
