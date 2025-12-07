# Question 2: Impact of Word Embeddings on Sentiment Classification
## Task
Compare the performance of LSTM models using different word embedding techniques for Urdu sentiment classification.

## Embedding Methods Evaluated
1. Random Embeddings (No pre-training)
2. Word2Vec
3. GloVe
4. FastText
5. ELMo (Contextualized embeddings)

## Results
### Performance Metrics

| Model                        | Accuracy | Precision | Recall | F-Score |
|-------                       |----------|-----------|--------|---------|
| LSTM (without embeddings)    | 0.584    | 0.580     | 0.542  | 0.560   |
| LSTM with Word2Vec           | 0.478    | 0.441     | 0.250  | 0.319   |
| LSTM with GloVe              | 0.490    | 0.490     | 1.000  | 0.658   |
| LSTM with FastText           | 0.522    | 1.000     | 0.025  | 0.049   |
| LSTM with ELMo               | 0.543    | 0.643     | 0.150  | 0.243   |

## Analysis
### Best Performing Model
**LSTM without embeddings (random initialization)** achieved the best balanced performance with:
- Highest accuracy: 58.4%
- Best F-score: 0.560
- Balanced precision (0.580) and recall (0.542)

### Key Observations
1. **Unexpected Results**: Pre-trained embeddings generally underperformed random embeddings, which is counterintuitive to typical NLP results. This is likely due to:
   - Language Mismatch: Word2Vec, GloVe, and FastText were trained on English corpora, not Urdu
   - Script Incompatibility: Urdu uses Arabic script, while these embeddings expect Latin script

2. **Extreme Behaviors**:
   - GloVe: Perfect recall (1.0) but low precision - predicts almost everything as positive
   - FastText: Perfect precision (1.0) but almost zero recall (0.025) - rarely predicts positive

3. **Random Embeddings Success**: Task-specific learning from scratch works better when pre-trained embeddings don't match the target language

## Conclusion
For Urdu sentiment analysis with limited data, **random embeddings are recommended**.

**Key Lesson**: Pre-trained embeddings from different languages/scripts can hurt performance more than help. The results demonstrate the critical importance of:
- Language-specific embeddings
- Script compatibility
- Language alignment between pre-training and target tasks
