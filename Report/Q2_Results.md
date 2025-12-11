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
|------------------------------|----------|-----------|--------|---------|
| LSTM (without embeddings)    | 0.596    | 0.596     | 0.542  | 0.568   |
| LSTM with Word2Vec           | 0.490*   | 0.490*    | 1.000* | 0.658*  |
| LSTM with GloVe              | 0.490*   | 0.490*    | 1.000* | 0.658*  |
| LSTM with FastText           | 0.490*   | 0.490*    | 1.000* | 0.658*  |
| LSTM with ELMo               | 0.543    | 0.536     | 0.492  | 0.513   |

*Note: Word2Vec, GloVe, and FastText all exhibit mode collapse (predicting all positive class)

## Analysis
### Best Performing Model
**LSTM without embeddings (random initialization)** achieved the best performance with:
- Highest accuracy: 59.6%
- Best F-score: 0.568
- Balanced precision (0.596) and recall (0.542)

### Critical Findings
1. **Widespread Mode Collapse**: Three out of four pre-trained embedding methods (Word2Vec, GloVe, FastText) completely collapsed into predicting only the positive class:
   - Perfect recall (1.0) but precision equals baseline (0.490)
   - All produce identical results despite different training algorithms
   - Models failed to learn discriminative features

2. **Why Pre-trained Embeddings Failed**:
   - **Custom Training on Small Data**: Word2Vec, GloVe, FastText were trained from scratch on ~700 Urdu sentences
   - **Insufficient Training Data**: Embeddings require thousands to millions of sentences to learn meaningful representations
   - **Poor Quality Embeddings**: Small corpus → poor statistical co-occurrence → weak embeddings
   - **Training Instability**: Low-quality embeddings confuse the LSTM, causing mode collapse

3. **ELMo Performance**: 
   - Only pre-trained embedding that didn't collapse (F1: 0.513)
   - Uses truly pre-trained contextualized representations (not trained from scratch)
   - Still underperforms random embeddings due to language/domain mismatch

4. **Random Embeddings Success**: 
   - Task-specific learning from scratch works better than poorly-trained embeddings
   - LSTM can learn task-relevant features directly during supervised training
   - No confusion from low-quality pre-computed representations

## Conclusion
For Urdu sentiment analysis with limited data, **random embeddings (no pre-training) are recommended** over custom-trained embeddings.

### Key Lessons
1. **Insufficient Data for Embedding Training**: 
   - Word2Vec, GloVe, FastText all failed when trained on ~700 sentences
   - Embedding methods require 10K-1M+ sentences for quality representations
   - Never train word embeddings from scratch on small datasets

2. **Mode Collapse Risk**: 
   - Poor quality embeddings cause more harm than random initialization
   - Multiple independent methods collapsed to identical predictions
   - Low-quality embeddings mislead the classifier

3. **Better Alternatives**:
   - **For small datasets**: Use random embeddings with supervised learning
   - **For production**: Use truly pre-trained multilingual models (Question 1 shows mBERT/XLM-RoBERTa achieve 70%+ accuracy)
   - **Never**: Train custom embeddings on datasets with < 10K sentences

4. **Comparison to Q1**: 
   - Q1's mBERT (70.6% accuracy) vs Q2's best (59.6% accuracy) = 11% improvement
   - Pre-trained transformer embeddings >> custom word embeddings
   - Transfer learning with proper pre-trained models is the only viable approach for low-resource Urdu NLP
