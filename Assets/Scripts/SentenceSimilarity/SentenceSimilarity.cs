using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.InferenceEngine;

namespace SentenceSimilarity.scripts
{
    public class SentenceSimilarity : MonoBehaviour
    {
        [Header("Assets")] public ModelAsset modelAsset;
        public TextAsset vocabAsset;

        [Header("Backend")] public BackendType backend = BackendType.GPUCompute;


        [Header("Tokenizer assumptions")]
        public int START_TOKEN = 101; // [CLS]
        public int END_TOKEN = 102; // [SEP]
        public bool useUNKFallback = true;

        [Header("Model output (hidden size)")]
        [Tooltip("Hidden size of your encoder (MiniLM-L6 = 384, BERT-base = 768)")]
        public int FEATURES = 384; // 384 for MiniLM-L6; 768 for BERT-base, etc.

        [Tooltip("Index of the encoder output that is the last hidden state (shape 1xNxh)")]
        public int modelLastHiddenStateOutputIndex = 0;

        private float[] _sentimentAxis;

        // --- internals ---
        private string[] _tokens;
        private Worker _engine;
        private Worker _dotScore;


        private void Start()
        {
            if (modelAsset == null)
            {
                Debug.LogError("Assign encoder ModelAsset.");
                return;
            }

            if (vocabAsset == null)
            {
                Debug.LogError("Assign vocab.txt TextAsset.");
                return;
            }

            _tokens = vocabAsset.text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);

            _engine = CreateMLModel();
            _dotScore = CreateDotScoreModel();
        }

        private void OnDestroy()
        {
            _dotScore?.Dispose();
            _engine?.Dispose();
        }

        public float CompareStrings(string a, string b)
        {
            var t1 = GetTokens(a);
            var t2 = GetTokens(b);
            using var e1 = GetEmbedding(t1);
            using var e2 = GetEmbedding(t2);
            return GetDotScore(e1, e2);
        }

        private float GetDotScore(Tensor<float> A, Tensor<float> B)
        {
            _dotScore.Schedule(A, B);
            var output = ((Tensor<float>)_dotScore.PeekOutput()).DownloadToNativeArray();
            return output[0];
        }


        private Tensor<float> GetEmbedding(List<int> tokenList)
        {
            int N = tokenList.Count;

            using var input_ids = new Tensor<int>(new TensorShape(1, N), tokenList.ToArray());
            using var token_type_ids = new Tensor<int>(new TensorShape(1, N), new int[N]);
            using var attention_mask = new Tensor<int>(new TensorShape(1, N), Enumerable.Repeat(1, N).ToArray());

            // Order must match your model's inputs (often: input_ids, attention_mask, token_type_ids)
            _engine.Schedule(input_ids, attention_mask, token_type_ids);

            // Last hidden state → mean pool (+L2 norm) handled inside engine graph
            var output = _engine.PeekOutput().ReadbackAndClone() as Tensor<float>;
            return output;
        }

        // ---------- Minimal WordPiece-like tokenizer (greedy) ----------
        private List<int> GetTokens(string text)
        {
            var words = text.ToLower().Split((char[])null, StringSplitOptions.RemoveEmptyEntries);

            var ids = new List<int> { START_TOKEN };
            var built = "";

            foreach (var word in words)
            {
                int start = 0;
                bool anyHit = false;

                for (int i = word.Length; i >= 0; i--)
                {
                    string piece = start == 0 ? word[..i] : "##" + word[start..i];
                    int index = Array.IndexOf(_tokens, piece);

                    if (index >= 0)
                    {
                        ids.Add(index);
                        built += piece + " ";
                        anyHit = true;

                        // full word matched
                        if (i == word.Length)
                        {
                            break;
                        }

                        start = i;
                        // restart search on the remainder
                        i = word.Length + 1;
                    }
                }

                if (!anyHit)
                {
                    int unk = Array.IndexOf(_tokens, "[UNK]");
                    ids.Add(unk >= 0 ? unk : 0);
                    built += "[UNK]";
                }
            }

            ids.Add(END_TOKEN);
            //Debug.Log("Tokenized: " + built);

            return ids;
        }

        private Worker CreateDotScoreModel()
        {
            FunctionalGraph g = new FunctionalGraph();
            FunctionalTensor a = g.AddInput<float>(new TensorShape(1, FEATURES));
            FunctionalTensor b = g.AddInput<float>(new TensorShape(1, FEATURES));
            FunctionalTensor dot = Functional.ReduceSum(a * b, 1); // (1,1)
            Model compiled = g.Compile(dot);
            return new Worker(compiled, backend);
        }

        private Worker CreateMLModel()
        {
            var model = ModelLoader.Load(modelAsset);

            // Wire the base model → take last hidden state
            var graph = new FunctionalGraph();
            var inputs = graph.AddInputs(model); // [input_ids, attention_mask, token_type_ids]
            var tokenEmbeddings =
                Functional.Forward(model, inputs)
                    [0]; // (1, N, FEATURES). If your model's output index differs, adjust here.
            var attentionMask = inputs[1];

            var pooled = MeanPooling(tokenEmbeddings, attentionMask);
            var compiled = graph.Compile(pooled);

            return new Worker(compiled, backend);
        }

        private FunctionalTensor MeanPooling(FunctionalTensor tokenEmbeddings, FunctionalTensor attentionMask)
        {
            FunctionalTensor mask = attentionMask.Unsqueeze(-1).BroadcastTo(new[] { FEATURES }); //shape=(1,N,FEATURES)
            FunctionalTensor sum = Functional.ReduceSum(tokenEmbeddings * mask, 1); //shape=(1,FEATURES)
            FunctionalTensor denom = Functional.ReduceSum(mask, 1) + 1e-9f; //shape=(1,FEATURES)
            FunctionalTensor mean = sum / denom;

            FunctionalTensor
                l2 = Functional.Sqrt(Functional.ReduceSum(Functional.Square(mean), 1, true)); //shape=(1,FEATURES)
            return mean / l2;
        }
    }
}