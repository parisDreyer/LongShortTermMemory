using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class Alphabet
    {
        public static string Letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
        public Dictionary<char, int> LettersByIndex;
        public Alphabet()
        {
            LettersByIndex = new Dictionary<char, int>();
            for (int i = 0; i < 26; i++)
                LettersByIndex.Add(Letters[i], i);
        }

        /// <summary>
        /// returns the 0-->25 letter index of a letter in the a->z roman alphabet.
        /// If the value is not a letter in roman alphabet returns 26
        /// </summary>
        /// <param name="letter"></param>
        /// <returns></returns>
        public int CorrespondingAlphabetIndex(char letter)
        {
            char vlu = char.ToUpper(letter);
            if (LettersByIndex.Keys.Contains(vlu))
                return LettersByIndex[vlu];
            else return 26;
        }
        public static double[] charToDouble(char L)
        {
            int idx = 0;
            char nL = Char.ToUpper(L);
            for (int i = 0; i < Letters.Length; ++i)
                if (Letters[i].Equals(nL)) idx = i;

            double[] retVl = new double[27];
            for (int r = 0; r < 27; ++r)
                if (idx == r) retVl[r] = 1.0;
            return retVl;
        }
        public static double[][][][] InitializeTrainingData(string[] words)
        {
            double[][][] Targ = new double[words.Length][][];
            double[][][] Inp = new double[words.Length][][];
            for (int wrds = 0; wrds < words.Length; ++wrds)
            {
                double[][][] tmp = stringToDoubles(words[wrds]);
                Inp[wrds] = tmp[0];
                Targ[wrds] = tmp[1];
            }
            return new double[][][][] { Inp, Targ };
        }
        public static double[][][] stringToDoubles(string word)
        {
            double[][] T = new double[word.Length][];
            double[][] I = new double[word.Length][];
            for (int C = 0; C < word.Length; ++C)
            {
                I[C] = Alphabet.charToDouble(word[C]);
                if (C < word.Length - 1) T[C] = Alphabet.charToDouble(word[C + 1]);
                else T[C] = new double[27];
            }
            return new double[][][] { I, T };
        }
    }
}
