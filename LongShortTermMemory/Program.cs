using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LongShortTermMemory
{
    class Program
    {
        static void Main(string[] args)
        {
            //Example.RNN.PredictLetters predict = new Example.RNN.PredictLetters(); predict.Example();
            //LSTMEXAMPLE lstm = new LSTMEXAMPLE(); lstm.Example();
            LSTM_TrivialExampl lstm = new LSTM_TrivialExampl(); lstm.Example();
            Console.ReadLine();
        }
    }
}
