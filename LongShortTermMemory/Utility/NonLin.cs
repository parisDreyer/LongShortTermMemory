using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class NonLin
    {
        /// <summary>
        /// hyperbolic tangent: output is between 1 and -1
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double HyperTan(double v)
        {
            if (v < -20.0) return -1.0;//approximation is correct to 30 decimals
            else if (v > 20.0) return 1.0;
            else return Math.Tanh(v);
        }
        /// <summary>
        /// output is between 0 and 1.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double LogisticSigmoid(double v)
        {
            return 1 / (1 + Math.Exp(-1.0 * v));//(Math.Exp(v)/(Math.Exp(v)+1));
        }
        public static double[] Softmax(double[] oSums)
        {
            //does all output nodes at once so scale
            //doesn't have to be recomputed each time
            double[] result = new double[oSums.Length];
            double sum = 0.0;
            for (int k = 0; k < oSums.Length; k++)
                sum += Math.Exp(oSums[k]);
            for (int k = 0; k < oSums.Length; k++)
                result[k] = Math.Exp(oSums[k]) / sum;
            return result; //now scaled so that vi sum to 1.0
        }

        /// <summary>
        /// the derivative of a point in an array of softmax outputs
        /// <para>this is the logistic sigmoid activation function. I just
        /// </para>made a separate method to have clear name differences
        /// Use this for output layer outputs.
        /// </summary>
        /// <param name="partial"></param>
        /// <returns></returns>
        public static double Softmax_Deriv(double partial)
        {
            return (1 - partial) * partial; // for softmax (same as log-sigmoid) with MSE
        }

        /// <summary>
        /// the derivative of the HyperBolic Tangent function. Use this for Hidden Layer Outputs
        /// </summary>
        /// <param name="partial"></param>
        /// <returns></returns>
        public static double TanH_Deriv(double partial)
        {
            return (1 + partial) * (1 - partial); // for tanh!
        }

        public static double LogisticSigmoid_Deriv(double partial)
        {
            return (LogisticSigmoid(partial) / (1 - LogisticSigmoid(partial)));
        }

        /// <summary>
        /// Performs a HyperbolicTangent or LogisticSigmoid ActivationFunction over a vector. if(derivative), performs the derivative function of the htan or logsig over the vector
        /// </summary>
        /// <param name="In"></param>
        /// <param name="logisticSigmoid"></param>
        /// <param name="derivative"></param>
        /// <returns></returns>
        public static double[] Activ(double[] In, bool logisticSigmoid = false, bool derivative = false)
        {
            double[] Out = new double[In.Length];
            if (derivative)
            {
                if (logisticSigmoid) for (int N = 0; N < In.Length; ++N) Out[N] = NonLin.LogisticSigmoid_Deriv(In[N]);
                else for (int N = 0; N < In.Length; ++N) Out[N] = NonLin.TanH_Deriv(In[N]);
            }
            else
            {
                if (logisticSigmoid) for (int N = 0; N < In.Length; ++N) Out[N] = NonLin.LogisticSigmoid(In[N]);
                else for (int N = 0; N < In.Length; ++N) Out[N] = NonLin.HyperTan(In[N]);
            }
            return Out;
        }
    }
}
