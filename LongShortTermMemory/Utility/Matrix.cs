using System;
using System.Collections.Generic;

namespace LongShortTermMemory
{
    /// <summary>
    /// Matrix Operations (should it be named 'Mop'?)
    /// </summary>
    class M
    {
        /// <summary>
        /// Multiplies each A[idx] by each element of B[idx]
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[][] ElMult(double[] A, double[][] B)
        {
            for (int i = 0; i < A.Length; i++)
                B[i] = M.ElMult(A[i], B[i]);
            return B;
        }
        /// <summary>
        /// Elementwise multiplication of Three vectors.
        /// returns 1xN dimension array where: N = A.Length <para>
        /// throws exception if A and B and C are different sizes </para>
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] ElMult(double[] A, double[] B, double[] C)
        {
            double[] retvlu = new double[A.Length];
            if (B.Length != A.Length || B.Length != C.Length) throw new IndexOutOfRangeException("The Arrays are different sizes");
            else for (int col = 0; col < retvlu.Length; ++col)
                    retvlu[col] = A[col] * B[col] * C[col];
            return retvlu;
        }
        /// <summary>
        /// Elementwise multiplication of two vectors.
        /// returns 1xN dimension array where: N = A.Length <para>
        /// throws exception if A and B are different sizes </para>
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] ElMult(double[] A, double[] B)
        {
            double[] retvlu = new double[A.Length];
            if (B.Length != A.Length) throw new IndexOutOfRangeException("The Arrays are different sizes");
            else for (int col = 0; col < retvlu.Length; ++col)
                    retvlu[col] = A[col] * B[col];
            return retvlu;
        }
        /// <summary>
        /// Multiplies each value in A by Coefficient C
        /// </summary>
        /// <param name="C"></param>
        /// <param name="A"></param>
        /// <returns></returns>
        public static double[] ElMult(double C, double[] A)
        {
            for (int col = 0; col < A.Length; ++col)
                    A[col] *= C;
            return A;
        }
        /// <summary>
        /// Multiplies each value in A by Coefficient C
        /// </summary>
        /// <param name="C"></param>
        /// <param name="A"></param>
        /// <returns></returns>
        public static double[][] ElMult(double C, double[][] A)
        {
            for (int row = 0; row < A.Length; ++row)
                for (int col = 0; col < A[row].Length; ++col)
                    A[row][col] *= C;
            return A;
        }
        /// <summary>
        /// Clips each value in A to a max decimal place value of maxPlace and rounds up to the smallest decimal place at minPlace (i.e. truncates last decimal place)
        /// </summary>
        /// <param name="A"></param>
        /// <param name="maxPlace"></param>
        /// <param name="minPlace"></param>
        /// <returns></returns>
        public static double[] Clip(double[] A, int maxPlace, int minPlace, double LearnRate)
        {
            for (int col = 0; col < A.Length; ++col)
            {
                A[col] = A[col] - Math.Round(A[col], maxPlace);
                A[col] = Math.Round(A[col], minPlace) * LearnRate;
            }

            return A;
        }
        /// <summary>
        /// Clips each value in A to less than [maxPlace] and rounds up to the [minPlace] decimal place (i.e. truncates last decimal place)
        /// </summary>
        /// <param name="A"></param>
        /// <param name="maxPlace"></param>
        /// <param name="minPlace"></param>
        /// <returns></returns>
        public static double[][] Clip(double[][] A, int maxPlace, int minPlace, double LearnRate)
        {
            for (int row = 0; row < A.Length; ++row)
                A[row] = M.Clip(A[row], maxPlace, minPlace, LearnRate);
            return A;
        }
        /// <summary>
        /// returns a set of randomly selected zeros and ones in an array of length(size)
        /// the total number of zeros is determined by the percent param
        /// this is meant to be used for the 'dropout' technique of backprop training
        /// </summary>
        /// <param name="size"></param>
        /// <param name="percent"></param>
        /// <param name="rnd"></param>
        /// <returns></returns>
        public static double[] Random_Dropout_Indices(int size, double percent, Random rnd)
        {//i got a little lazy so started using lists in this method instead of arrays as per the rest of the project
            double[] retVal = new double[size];
            int numDrop = (int)Math.Round(percent * (double)size);
            List<int> unselected_indices = new List<int>(size);
            for (int s = 0; s < size; ++s) //indices for potential 0 coefficient
                unselected_indices.Add(s);
            List<int> dropIndices = new List<int>(numDrop); //the indices of values to drap
            for (int i = 0; i < size; ++i)
            {
                if (numDrop > 0)
                {
                    int randomIDX = rnd.Next(0, unselected_indices.Count); //the index of the index to drop
                    int drop_idx = unselected_indices[randomIDX]; //the index to drop
                    unselected_indices.RemoveAt(randomIDX); //remove the index to drop from the list of indices to drop
                    dropIndices.Add(drop_idx); // set the value of the drop index
                    numDrop -= 1;
                }
                else
                    break;
            }
            for (int s = 0; s < size; ++s) //set the return value Array to 0s and 1s
                if (dropIndices.Contains(s)) //if randomly selected index is a 'set to zero' index
                    retVal[s] = 0;
                else //else set to a 1 coefficient as normal
                    retVal[s] = 1;
                    
            return retVal;
        }

        /// <summary>
        /// Outer-Product of two Vectors. 
        /// returns a double[A.Len][B.Len] where each column is all of A[each]*B[col]
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[][] Outer(double[] A, double[] B)
        {
            double[][] V = M.MakeMatrix(A.Length, B.Length);
            for (int bCol = 0; bCol < B.Length; ++bCol)
                for (int aEach = 0; aEach < A.Length; ++aEach)
                    V[aEach][bCol] = A[aEach] * B[bCol];
            return V;
        }


        /// <summary>
        /// Output V is B[0].Length
        /// Each col of V = A * Each row of B
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] Dot(double[] A, double[][] B)
        {
            if (A.Length != B.Length) throw new IndexOutOfRangeException("Matrix size is incompatible");
            double[] V = new double[B[0].Length];
            for (int col = 0; col < B[0].Length; ++col)
                for (int row = 0; row < A.Length; ++row)
                    V[col] += A[row] * B[row][col];
            return V;
        }

        /// <summary>
        /// Output V is A.Length, as a 1xA[0].Len Array (as if it were an A[0].Len x 1 Array)
        /// Each col of V = A[row][each] * B[each], treating B as if it were a single column vector
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] Dot(double[][] A, double[] B)
        {
            if (A[0].Length != B.Length) throw new IndexOutOfRangeException("Matrix size is incompatible");
            double[] V = new double[A.Length];
            for (int row = 0; row < A.Length; ++row)
                for(int col = 0; col<A[0].Length;++col)
                    V[row] += A[row][col] * B[col];
            return V;
        }

        /// <summary>
        /// Rows become the columns and columns become the rows
        /// </summary>
        /// <param name="M"></param>
        /// <returns></returns>
        public static double[][] T(double[][] M)
        {
            double[][] T = LongShortTermMemory.M.MakeMatrix(M[0].Length, M.Length);
            for (int col = 0; col < M[0].Length; ++col)
                for (int row = 0; row < M.Length; ++row)
                    T[col][row] = M[row][col];
            return T;
        }
        /// <summary>
        /// Rows become the columns and columns become the rows
        /// </summary>
        /// <param name="M"></param>
        /// <returns></returns>
        public static double[][] T(double[] M)
        {
            double[][] T = LongShortTermMemory.M.MakeMatrix(M.Length, M.Length);
            for (int col = 0; col < M.Length; ++col)
                for (int row = 0; row < M.Length; ++row)
                    T[col][row] = M[col];
            return T;
        }
        /// <summary>
        /// sums two vectors columnwise, if the two vectors are of equal length
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] Sum(double[] A, double[] B)
        {
            double[] retvlu = new double[A.Length];
            if (retvlu.Length != B.Length) throw new IndexOutOfRangeException("Matrix size is incompatible");
            for (int idx = 0; idx < retvlu.Length; ++idx)
                retvlu[idx] = A[idx] + B[idx];
            return retvlu;
        }

        /// <summary>
        /// sums rows columnwise. Matrix summation is commutative so row order not matter
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[] Sum(double[][] toSumRows)
        {
            double[] retvlu = new double[toSumRows[0].Length];
            for (int row = 0; row < toSumRows.Length; ++row)//foreach row
                for (int col = 0; col < toSumRows[0].Length; ++col)//foreach column
                    retvlu[col] += toSumRows[row][col];
            return retvlu;
        }
        /// <summary>
        /// Sums each value in matrices A and B element-wise
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static double[][] Sum(double[][] A, double[][] B)
        {
            double[][] V = M.MakeMatrix(A.Length, A[0].Length);
            for (int R = 0; R < A.Length; ++R)
                V[R] = Sum(A[R], B[R]);
            return V;
        }
        public static double[][] MakeJaggedMatrix(int[] columnsAtlayers)
        {
            double[][] retValue = new double[columnsAtlayers.Length][];
            for (int row = 0; row < columnsAtlayers.Length; row++)
                retValue[row] = new double[columnsAtlayers[row]];
            return retValue;
        }
        public static double[][] MakeSquareMatrix(int nColumns)
        {
            double[][] retValue = new double[nColumns][];
            for (int row = 0; row < nColumns; row++)
                retValue[row] = new double[nColumns];
            return retValue;
        }
        public static double[][] MakeMatrix(int nRows, int nColumns)
        {
            double[][] retValue = new double[nRows][];
            for (int row = 0; row < nRows; row++)
                retValue[row] = new double[nColumns];
            return retValue;
        }
        /// <summary>
        /// can use this for Weights or Biases. Biases can be thought of as a weight that you add to the the activated-weighted output rather than multiply
        /// </summary>
        /// <param name="Weights"></param>
        /// <param name="numberOfWeights"></param>
        /// <param name="rndm"></param>
        /// <returns></returns>
        public static double[] InitializeWeights(int numberOfWeights, Random rndm)
        {
            double[] Weights = new double[numberOfWeights];
            //weights            
            double hi = 0.10;
            double lo = -0.10;
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = (hi - lo) * rndm.NextDouble() + lo;
            return Weights;
        }
    }
}
