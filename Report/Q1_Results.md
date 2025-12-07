# Question 1: Sentiment Analysis on Urdu Tweets
## Task
Perform sentiment analysis on Urdu tweets using different neural network architectures and compare with multilingual pre-trained models.

## Models Evaluated
1. RNN (Recurrent Neural Network)
2. GRU (Gated Recurrent Unit)
3. LSTM (Long Short-Term Memory)
4. BiLSTM (Bidirectional LSTM)
5. mBERT (Multilingual BERT)
6. XLM-RoBERTa (Cross-lingual Language Model)

## Results
### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RNN   | 0.531    | 0.514     | 0.792  | 0.623    |
| GRU   | 0.551    | 0.554     | 0.425  | 0.481    |
| LSTM  | 0.567    | 0.592     | 0.375  | 0.459    |
| BiLSTM| 0.571    | 0.555     | 0.633  | 0.591    |
| mBERT | 0.678    | 0.847     | 0.417  | 0.559    |
|XLM-RoBERTa|0.714 | 0.705     | 0.717  | 0.711    |

## Analysis
### Best Performing Model
**XLM-RoBERTa** achieved the best overall performance with:
- Highest accuracy: 71.4%
- Best F1-score: 0.711
- Balanced precision (0.705) and recall (0.717)

### Key Observations
1. **Pre-trained Models Superiority**: Both mBERT and XLM-RoBERTa significantly outperformed traditional RNN-based models, demonstrating the value of transfer learning for low-resource languages like Urdu.

2. **XLM-RoBERTa vs mBERT**: XLM-RoBERTa outperformed mBERT by 3.6% in accuracy, likely due to:
   - Better cross-lingual representations
   - RoBERTa's improved training methodology

3. **RNN-based Models**: Among traditional models, RNN surprisingly achieved the best F1-score (0.623), though with lower precision. BiLSTM showed the most balanced performance among RNN variants.

## Conclusion
For Urdu sentiment analysis, **XLM-RoBERTa is the recommended model** due to its:
- Superior overall performance
- Balanced precision-recall trade-off
- Better cross-lingual understanding
- Strong performance on low-resource languages