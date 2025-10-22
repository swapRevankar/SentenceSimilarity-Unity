using System;
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.UIElements;

namespace SentenceSimilarity.scripts
{
    /// <summary>
    /// Simple UI driver for RunMiniLM:
    /// - Takes two TMP_InputFields
    /// - Shows cosine similarity text
    /// - Fills a bar from 0..1 and changes color
    /// </summary>
    public class SentenceSimilarityUI : MonoBehaviour
    {
        [SerializeField] private VisualTreeAsset mainLayout; // SentenceSimilarity.uxml
        [SerializeField] private VisualTreeAsset rowTemplate; // SentenceRow.uxml
        [SerializeField] private VisualTreeAsset scoreTemplate; // SentenceScore.uxml

        public SentenceSimilarity runner;

        private TextField _sourceInput;
        private ScrollView _sentencesContainer;
        private Button _addSentenceButton;
        private Button _computeButton;
        private ScrollView _resultsContainer;

        private readonly List<TemplateContainer> _candidates = new();


        private void OnEnable()
        {
            var root = GetComponent<UIDocument>().rootVisualElement;
            mainLayout.CloneTree(root);

            _sourceInput = root.Q<TextField>("SourceInput");
            _sentencesContainer = root.Q<ScrollView>("SentencesContainer");
            _addSentenceButton = root.Q<Button>("AddSentence");
            _computeButton = root.Q<Button>("ComputeButton");
            _resultsContainer = root.Q<ScrollView>("ResultsContainer");

            _addSentenceButton.clicked += AddSentenceRow;
            _computeButton.clicked += ComputeSimilarity;

            AddSentenceRow();
        }

        private void AddSentenceRow()
        {
            var row = rowTemplate.CloneTree();
            _sentencesContainer.Add(row);
            _candidates.Add(row);
        }

        private void ComputeSimilarity()
        {
            string sourceSentence = _sourceInput.value?.Trim();

            _resultsContainer.Clear();

            foreach (var candidate in _candidates)
            {
                var input = candidate.Q<TextField>("CompareInput");
                var text = input.value?.Trim();

                var score = runner.CompareStrings(sourceSentence, text);

                var row = scoreTemplate.CloneTree();

                var sentenceLabel = row.Q<Label>("SentenceLabel");
                var scoreLabel = row.Q<Label>("ScoreLabel");
                var barBg = row.Q<VisualElement>("BarBg");
                var barFill = barBg?.Q<VisualElement>("BarFill");

                if (sentenceLabel != null)
                {
                    sentenceLabel.text = text;
                }

                if (scoreLabel != null)
                {
                    scoreLabel.text = score.ToString("0.000");
                }

                if (barFill != null)
                {
                    // float percent = Mathf.Clamp01(score) * 100f;
                    // barFill.style.width = Length.Percent(percent);
                    StartCoroutine(AnimateBarFill(barFill, score));
                }

                _resultsContainer.Add(row);
            }
        }


        private IEnumerator AnimateBarFill(VisualElement bar, float score)
        {
            Color startColor = new Color(0.6f, 0.4f, 1f);
            Color endColor   = new Color(0.5f, 0.7f, 1f);

            Color col = Color.Lerp(startColor, endColor, score);
            bar.style.backgroundColor = new StyleColor(col);

            float duration = 0.5f;
            float t = 0f;
            float start01 = 0f;

            while (t < duration)
            {
                t += Time.deltaTime;
                float k = Mathf.SmoothStep(0f, 1f, t / duration);
                float v = Mathf.Lerp(start01, score, k);
                bar.style.width = Length.Percent(v * 100f);
                yield return null;
            }

            bar.style.width = Length.Percent(score * 100f);
        }


    }
}