# Question 4: Impact of Pre-trained Embeddings on Translation
## Task
Compare RNN Seq2Seq models using random embeddings vs. pre-trained GloVe embeddings for English-to-Urdu translation.

## Models Evaluated
1. RNN Seq2Seq with Random Embeddings
2. RNN Seq2Seq with GloVe Embeddings

## Results
### Performance Comparison

| Model             | BLEU Score | Final Val Loss | Training Time (s) | Avg Time/Epoch (s) |
|-------            |------------|----------------|-------------------|------------------- |
| Random Embeddings | 18.996     | 5.390          | 60.25             | 0.301              |
| GloVe Embeddings  | 18.996     | 5.475          | 53.89             | 0.269              |

### Sample Translations
Both models produce nearly identical outputs showing severe mode collapse:

#### Example: "return next week"
- Reference: اگلے ہفتے واپس آؤ
- Random: وہ ایک استاد ہے (he is a teacher)
- GloVe: وہ ایک ہے ہے (he one is is)

#### Example: "i eat rice"
- Reference: میں چاول کھاتا ہوں
- Random: وہ ایک استاد ہے
- GloVe: وہ ایک ہے ہے

#### Example: "we play cricket"
- Reference: ہم کرکٹ کھیلتے ہیں
- Random: وہ ایک استاد ہے
- GloVe: وہ ایک ہے ہے

## Analysis
### Key Findings
1. **No Meaningful Difference**:
   - Identical BLEU scores (18.996)
   - Similar validation losses (~5.4)
   - Both produce repetitive, meaningless outputs

2. **GloVe Provides No Benefit**:
   - Despite using pre-trained embeddings, performance doesn't improve
   - Only advantage: 10% faster training (53.9s vs 60.2s)

3. **Mode Collapse in Both Models**:
   - Random model: Always outputs "وہ ایک استاد ہے"
   - GloVe model: Always outputs "وہ ایک ہے ہے"

### Why GloVe Failed to Help
1. **Language Mismatch**: GloVe embeddings were trained on English text, but target language is Urdu (different script, grammar)

2. **Insufficient Data Dominates**: With only ~82 training samples, data limitation overwhelms any embedding advantage

3. **Decoder Bottleneck**: Urdu decoder has no pre-training, and cross-lingual alignment can't be learned from 82 samples

## Conclusion
Pre-trained embeddings provide **no translation quality benefit** with extremely limited data.

### Recommendations for English-Urdu Translation
1. **Data First**: Collect 10,000+ parallel sentences minimum
2. **Use Pre-trained Models**: mBART, mT5, or NLLB (as shown in Q3)
3. **Avoid training from scratch** regardless of embedding choice

**Key Lesson**: With only 82 training samples, embedding initialization is irrelevant - the fundamental problem is data scarcity, not initialization strategy.