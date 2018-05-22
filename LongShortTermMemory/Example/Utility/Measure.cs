using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class Measure
    {
        public static double CrossEntropyError(double[] output, double[] target)
        {
            double Loss = 0.0;
            for (int idx = 0; idx < output.Length; ++idx)
                Loss += (-1.0) * Math.Log(output[idx]) * target[idx];
            return Loss;
        }

        /// <summary>
        /// Use instead of derivatives for crossentropy error
        /// returns an array of doubles with elementwise subtraction of T-O
        /// </summary>
        /// <param name="T"></param>
        /// <param name="O"></param>
        /// <returns></returns>
        public static double[] TminusO(double[] T, double[] O)
        {
            double[] V = new double[T.Length];
            for (int idx = 0; idx < T.Length; ++idx)
                V[idx] = T[idx] - O[idx];
            return V;
        }

        /// <summary>
        /// returns a randomized order of index values that can be used to select from an ordered array of values
        /// </summary>
        /// <param name="sequenceLength"></param>
        /// <param name="rnd"></param>
        /// <returns></returns>
        public static int[] Shuffle(int sequenceLength, Random rnd) // instance method
        {
            int[] sequence = new int[sequenceLength];
            for (int s = 0; s < sequenceLength; s++)
                sequence[s] = s;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
            return sequence;
        } // Shuffle
          /// <summary>
          /// returns the index of the largest two values in a array of double values
          /// </summary>
          /// <param name="values"></param>
          /// <returns></returns>
        public static int[] MaxIndex(double[] values)
        {
            int[] lrgstVlus = new int[2];
            double LrgstVlu = 0.0;
            double scndLrgst = 0.0;
            for (int idx = 0; idx < values.Length; idx++)
                if (values[idx] > LrgstVlu)
                {
                    scndLrgst = LrgstVlu;
                    LrgstVlu = values[idx];
                    lrgstVlus[1] = lrgstVlus[0];//first lrgst prvious becomes second largest
                    lrgstVlus[0] = idx;//first largest gets set to new largest
                }
            return lrgstVlus;
        }
    }
}
