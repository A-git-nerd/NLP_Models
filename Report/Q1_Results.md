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
| RNN   | 0.494    | 0.485     | 0.550  | 0.516    |
| GRU   | 0.567    | 0.564     | 0.517  | 0.539    |
| LSTM  | 0.490*   | 0.490*    | 1.000* | 0.658*   |
| BiLSTM| 0.567    | 0.555     | 0.592  | 0.573    |
| mBERT | 0.706    | 0.671     | 0.783  | 0.723    |
|XLM-RoBERTa|0.698 | 0.642     | 0.867  | 0.738    |

*Note: LSTM shows mode collapse (predicting all positive class)

## Analysis
### Best Performing Model
**XLM-RoBERTa** achieved the best overall performance with:
- Best F1-score: 0.738
- High recall: 0.867
- Accuracy: 69.8%

**mBERT** achieved the highest accuracy (70.6%) with more balanced precision-recall.

### Key Observations
1. **Pre-trained Models Superiority**: Both mBERT and XLM-RoBERTa significantly outperformed traditional RNN-based models (20%+ improvement in F1), demonstrating the critical value of transfer learning for low-resource languages like Urdu.

2. **XLM-RoBERTa vs mBERT**: 
   - XLM-RoBERTa has better F1 (0.738 vs 0.723) and much higher recall (0.867 vs 0.783)
   - mBERT has slightly higher accuracy (70.6% vs 69.8%) and precision
   - XLM-RoBERTa's superior cross-lingual representations make it better for comprehensive sentiment detection

3. **RNN-based Models Performance**:
   - BiLSTM: Best among RNN variants (F1: 0.573, Accuracy: 56.7%)
   - GRU: Similar to BiLSTM (F1: 0.539, Accuracy: 56.7%)
   - **LSTM: Mode collapse issue** - predicts all positive class (perfect recall, low precision)
   - RNN: Weakest performance (F1: 0.516, Accuracy: 49.4%)

4. **LSTM Stability Issues**: Unidirectional LSTM suffered from mode collapse despite hyperparameter tuning and architectural improvements. BiLSTM's bidirectional context provides better gradient flow and stability.

## Conclusion
For Urdu sentiment analysis, **XLM-RoBERTa is the recommended model** due to its:
- Best F1-score (0.738) and highest recall (0.867)
- Superior cross-lingual understanding
- Strong performance on low-resource languages
- Better generalization than mBERT for sentiment detection

### Key Takeaways
1. **Transfer learning is essential**: ~20% F1 improvement over best traditional model
2. **Architecture matters**: BiLSTM significantly more stable than unidirectional LSTM
3. **Pre-trained multilingual models** (mBERT, XLM-RoBERTa) are mandatory for production Urdu NLP
4. **Traditional RNNs** struggle with low-resource scenarios and are prone to training instabilities