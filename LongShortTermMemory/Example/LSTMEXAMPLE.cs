using System;
namespace LongShortTermMemory
{
    class LSTMEXAMPLE
    {
        private System.DateTime PROCESS_START;
        public LSTMEXAMPLE() {
            PROCESS_START = DateTime.Now;
            Console.WriteLine("LSTM Example");
        }
        public void Example()
        {
            Random rndm = new Random(0);
            int nLayers = 1;
            int nIn = 27;
            int nOut = nIn;
            int memCellCount = 1000;//256;
            int nT = 20;//50;
            double LR = 0.001;

            LSTMRNN NET = new LSTMRNN(nIn,memCellCount,nOut,nLayers,nT,rndm);
            string[] vocab = new string[] 
            { " squid flax"," zephyr tow", " not the droids",
                " you are"," looking for", " jewel in", " the heart", " of the lotus", " blimp swum", " berg pyramid",
                " I learned this", "at least",
                " by my experiment that if one advances confidently in the direction of his or her dreams",
                " and endeavours to live the life which he or she has imagined",
                " he will meet with a success unexpected in common hours", " He will put some things behind",
                " will pass an invisible boundary", " new", " universal",
                " and more liberal laws will begin to establish themselves around and within him",
                " or the old laws be expanded", " and interpreted in his favour in a more liberal sense",
                " and he will live with the license of a higher order of beings",
                " In proportion as he simplifies his life",
                " the laws of the universe will appear less complex",
                " and solitude will not be solitude",
                " nor poverty poverty",
                " nor weakness weakness", 
                " If you have built castles in the air",
                " your work need not be lost",
                " that is where they should be",
                " Now put the foundations under them",
                " i success published in a masque,","success published in a masque of",
            " published in a masque of poets",
" in a masque of poets at",
" a masque of poets at the",
" masque of poets at the request",
" of poets at the request of",
" poets at the request of",
" at the request of hh the",
" the request of hh the authors",
" request of hh the authors fellowtownswoman",
" of the authors fellowtownswoman and",
" the authors fellowtownswoman and friend",
" the authors fellowtownswoman and friend success",
" authors fellowtownswoman and friend success is",
" fellowtownswoman and friend success is counted",
" and friend success is counted sweetest",
" friend success is counted sweetest by",
" success is counted sweetest by those",
" is counted sweetest by those who",
" counted sweetest by those who neer",
" sweetest by those who neer succeed",
" by those who neer succeed to",
" those who neer succeed to comprehend",
" who neer succeed to comprehend a",
" neer succeed to comprehend a nectar",
" succeed to comprehend a nectar requires",
" to comprehend a nectar requires sorest",
" comprehend a nectar requires sorest need",
" a nectar requires sorest need not",
" nectar requires sorest need not one",
" requires sorest need not one of",
" sorest need not one of all",
" need not one of all the",
" not one of all the purple",
" one of all the purple host",
" of all the purple host who",
" all the purple host who took",
" the purple host who took the",
" purple host who took the flag",
" host who took the flag today",
" who took the flag today can",
" took the flag today can tell",
" the flag today can tell the",
" flag today can tell the definition",
" today can tell the definition so",
" can tell the definition so clear",
" tell the definition so clear of",
" the definition so clear of victory",
" definition so clear of victory as",
" so clear of victory as he",
" clear of victory as he defeated",
" of victory as he defeated dying",
" victory as he defeated dying on",
" as he defeated dying on whose",
" he defeated dying on whose forbidden",
" defeated dying on whose forbidden ear",
" dying on whose forbidden ear the",
" on whose forbidden ear the distant",
" whose forbidden ear the distant strains",
" forbidden ear the distant strains of",
" ear the distant strains of triumph",
" the distant strains of triumph break",
" distant strains of triumph break agonized",
" strains of triumph break agonized and",
" of triumph break agonized and clear",
" triumph break agonized and clear",
" break agonized and clear our",
" agonized and clear our share",
" and clear our share of",
"  our share of night to",
            };
            int maxepoch = 4000;
            int iter = 0;
            double[] output = new double[nOut];
            double[][][][] TrainingData = Alphabet.InitializeTrainingData(vocab);
            while (iter < maxepoch)
            {
                int[] S = Measure.Shuffle(TrainingData[0].Length, rndm);
                for (int W = 0; W < TrainingData[0].Length; ++W)
                {
                    for (int L = 0; L < TrainingData[0][S[W]].Length; ++L)
                    {
                        Array.Copy(NET.ComputeOutputs(TrainingData[0][S[W]][L]), output, nOut);
                        NET.ComputeInputError(Measure.TminusO(TrainingData[1][S[W]][L], output), LR, rndm,1, 10, 5);
                        Console.Write(Alphabet.Letters[Measure.MaxIndex(output)[0]]); // write values
                    }
                }
                READOUT(iter, nOut);
                ++iter;
            }
        }
        private void READOUT(int T,int nOut)
        {
            //set color!
            ConsoleColor prvTxtClr = Console.ForegroundColor;
            ConsoleColor prvBackgrndClr = Console.BackgroundColor;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.Yellow;
            //Write values!
            Console.WriteLine("\nITERATION: {0} \nTIME: [START: {1}], [NOW: {2}]",T, PROCESS_START, DateTime.Now);
            //set color back!
            Console.ForegroundColor = prvTxtClr;
            Console.BackgroundColor = prvBackgrndClr;
        }
    }
}
