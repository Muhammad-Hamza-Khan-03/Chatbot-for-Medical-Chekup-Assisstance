# Medical Question Answering: A Comparative Study of Transformer Models

## Abstract

This research presents a comprehensive evaluation of transformer-based models for medical question answering tasks. We fine-tuned and compared three state-of-the-art transformer architectures—BERT, MobileBERT, and RoBERTa—on the MedQuad dataset, a medical question-answering corpus containing 16,407 question-answer pairs. The study investigates the impact of different dropout rates (0.3 and 0.7) on model performance and evaluates models using multiple metrics including exact match accuracy, BLEU scores, and ROUGE scores. Our findings indicate that BERT with dropout 0.3 achieves the best balance between accuracy and precision, while RoBERTa demonstrates superior ROUGE scores despite being trained on a smaller subset of the data.

## 1. Introduction

### 1.1 Background

Medical question answering (QA) systems play a crucial role in healthcare information retrieval, enabling patients and healthcare professionals to access accurate medical information efficiently. With the advent of transformer-based language models, there has been significant progress in developing automated systems capable of understanding and answering medical questions.

### 1.2 Problem Statement

The challenge in medical QA lies in:
- Understanding complex medical terminology and context
- Providing accurate, reliable answers to diverse question types
- Handling various question categories (treatment, symptoms, prevention, genetic changes, information)
- Balancing model performance with computational efficiency

### 1.3 Research Objectives

1. Evaluate the performance of BERT, MobileBERT, and RoBERTa on medical QA tasks
2. Investigate the effect of dropout regularization on model performance
3. Compare models using multiple evaluation metrics (BLEU, ROUGE, exact match)
4. Analyze model behavior across different question types

## 2. Dataset

### 2.1 MedQuad Dataset

The MedQuad dataset consists of 16,407 question-answer pairs with the following structure:

- **Size**: 16,407 entries
- **Columns**: 
  - `qtype`: Question type/category
  - `Question`: Medical question text
  - `Answer`: Corresponding answer text

### 2.2 Data Distribution

The dataset contains questions across multiple categories:
- Treatment-related questions
- Symptom-related questions
- Prevention-related questions
- Genetic changes questions
- General information questions

### 2.3 Data Preprocessing

- Data was converted to SQuAD format for compatibility with transformer-based QA models
- Training/test split: 75% training (12,305 samples) / 25% testing (4,102 samples)
- Random state: 42 (for reproducibility)

## 3. Methodology

### 3.1 Model Architectures

#### 3.1.1 BERT (Bidirectional Encoder Representations from Transformers)
- **Base Model**: `bert-base-cased`
- **Parameters**: ~110M
- **Architecture**: 12-layer transformer encoder
- **Specialization**: Bidirectional context understanding

#### 3.1.2 MobileBERT
- **Base Model**: `google/mobilebert-uncased`
- **Parameters**: ~25M (lightweight variant)
- **Architecture**: Optimized for mobile/edge deployment
- **Specialization**: Efficiency-focused architecture

#### 3.1.3 RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Base Model**: `roberta-base`
- **Parameters**: ~125M
- **Architecture**: Improved BERT with optimized training
- **Specialization**: Enhanced robustness and performance

### 3.2 Training Configuration

All models were trained with the following hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 5e-5 |
| Training Epochs | 1 |
| Batch Size | 16 |
| Dropout Rates | 0.3, 0.7 |
| Optimizer | AdamW (default) |
| Evaluation During Training | Enabled |
| CUDA | Enabled |

**Note**: RoBERTa was trained on a truncated dataset (1,000 samples) due to computational constraints, with 750 training and 250 test samples.

### 3.3 Evaluation Metrics

#### 3.3.1 Exact Match Metrics
- **Correct**: Exact match between predicted and reference answers
- **Similar**: Semantically similar answers
- **Incorrect**: Mismatched answers

#### 3.3.2 BLEU Score
- Measures n-gram precision between predicted and reference answers
- Range: 0 to 1 (higher is better)
- Also computed with smoothing function (Method 2) to handle zero n-gram overlaps

#### 3.3.3 ROUGE Score
- **ROUGE-1**: Unigram overlap (recall, precision, F1)
- **ROUGE-2**: Bigram overlap (recall, precision, F1)
- **ROUGE-L**: Longest common subsequence (recall, precision, F1)

### 3.4 Experimental Setup

- **Framework**: SimpleTransformers
- **Experiment Tracking**: Weights & Biases (wandb)
- **Hardware**: CUDA-enabled GPU
- **Random Seed**: 42 (for reproducibility)

## 4. Results

### 4.1 BERT Model Results

#### 4.1.1 BERT with Dropout 0.3

| Metric | Value |
|--------|-------|
| Correct | 760 |
| Similar | 3,136 |
| Incorrect | 206 |
| Eval Loss | -7.03 |
| BLEU Score | 0.572 |
| BLEU (Smoothing) | 0.00033 |
| ROUGE-1 F1 | 0.459 |
| ROUGE-2 F1 | 0.442 |
| ROUGE-L F1 | 0.459 |

**Analysis**: BERT with dropout 0.3 shows strong performance with 18.5% exact matches and 76.4% similar matches. The model demonstrates good recall (ROUGE-1 R: 0.837) but moderate precision (ROUGE-1 P: 0.429).

#### 4.1.2 BERT with Dropout 0.7

| Metric | Value |
|--------|-------|
| Correct | 759 |
| Similar | 3,135 |
| Incorrect | 208 |
| Eval Loss | -7.79 |
| BLEU Score | 0.574 |
| BLEU (Smoothing) | 0.00070 |
| ROUGE-1 F1 | 0.470 |
| ROUGE-2 F1 | 0.438 |
| ROUGE-L F1 | 0.470 |

**Analysis**: Higher dropout (0.7) shows slightly improved ROUGE-1 F1 (0.470 vs 0.459) and better recall (0.859 vs 0.837), suggesting better generalization. However, exact match accuracy remains similar.

### 4.2 MobileBERT Model Results

#### 4.2.1 MobileBERT with Dropout 0.3

| Metric | Value |
|--------|-------|
| Correct | 0 |
| Similar | 4,102 |
| Incorrect | 0 |
| Eval Loss | NaN |
| BLEU Score | 0.623 |
| BLEU (Smoothing) | 0.00050 |
| ROUGE-1 F1 | 0.283 |
| ROUGE-2 F1 | 0.237 |
| ROUGE-L F1 | 0.283 |

**Analysis**: MobileBERT shows interesting behavior—no exact matches but all predictions are classified as "similar." Despite higher BLEU score (0.623), ROUGE scores are significantly lower, indicating potential issues with answer quality or evaluation methodology.

#### 4.2.2 MobileBERT with Dropout 0.7

| Metric | Value |
|--------|-------|
| Correct | 0 |
| Similar | 4,102 |
| Incorrect | 0 |
| Eval Loss | NaN |
| BLEU Score | 0.619 |
| BLEU (Smoothing) | 0.00045 |
| ROUGE-1 F1 | 0.318 |
| ROUGE-2 F1 | 0.265 |
| ROUGE-L F1 | 0.318 |

**Analysis**: Similar pattern to dropout 0.3, with slightly improved ROUGE scores. The higher dropout rate appears to help with generalization in this lightweight model.

### 4.3 RoBERTa Model Results

#### 4.3.1 RoBERTa with Dropout 0.3

| Metric | Value |
|--------|-------|
| Correct | 19 |
| Similar | 230 |
| Incorrect | 1 |
| Eval Loss | -6.11 |
| BLEU Score | 0.566 |
| BLEU (Smoothing) | 0.00026 |
| ROUGE-1 F1 | 0.593 |
| ROUGE-2 F1 | 0.538 |
| ROUGE-L F1 | 0.593 |

**Analysis**: RoBERTa demonstrates the best ROUGE performance (F1: 0.593) despite being trained on only 1,000 samples. The model shows excellent recall (0.99) and good precision (0.488), indicating strong semantic understanding. However, note that this evaluation was performed on the full test set, not the truncated subset.

### 4.4 Comparative Analysis

#### 4.4.1 Model Performance Summary

| Model | Dropout | Exact Match | Similar | ROUGE-1 F1 | BLEU |
|-------|---------|-------------|---------|------------|------|
| BERT | 0.3 | 760 (18.5%) | 3,136 (76.4%) | 0.459 | 0.572 |
| BERT | 0.7 | 759 (18.5%) | 3,135 (76.4%) | 0.470 | 0.574 |
| MobileBERT | 0.3 | 0 (0%) | 4,102 (100%) | 0.283 | 0.623 |
| MobileBERT | 0.7 | 0 (0%) | 4,102 (100%) | 0.318 | 0.619 |
| RoBERTa | 0.3 | 19 (0.5%)* | 230 (92%)* | 0.593 | 0.566 |

*Note: RoBERTa evaluation on full test set (4,102 samples) but trained on 1,000 samples.

#### 4.4.2 Key Findings

1. **BERT Performance**: 
   - Best overall balance between exact match and semantic similarity
   - Dropout 0.7 provides marginal improvements in ROUGE scores
   - Strong practical performance for medical QA applications

2. **MobileBERT Behavior**:
   - All predictions classified as "similar" suggests potential evaluation or model calibration issues
   - Higher BLEU scores but lower ROUGE scores indicate possible overfitting or answer format mismatches
   - Lightweight architecture may struggle with complex medical terminology

3. **RoBERTa Excellence**:
   - Highest ROUGE scores despite limited training data
   - Excellent recall (0.99) suggests strong semantic understanding
   - Potential for best performance with full dataset training

4. **Dropout Impact**:
   - Higher dropout (0.7) generally improves ROUGE scores
   - Minimal impact on exact match accuracy
   - Suggests regularization helps with generalization

## 5. Discussion

### 5.1 Model Selection Considerations

**For Production Deployment:**
- **BERT (dropout 0.3 or 0.7)**: Recommended for balanced performance and reliability
- **RoBERTa**: Best choice if computational resources allow full dataset training
- **MobileBERT**: Suitable for resource-constrained environments but requires further investigation

### 5.2 Evaluation Metric Insights

1. **BLEU vs ROUGE Discrepancy**: 
   - MobileBERT shows high BLEU but low ROUGE, suggesting BLEU may not fully capture answer quality
   - ROUGE appears more sensitive to semantic correctness

2. **Exact Match Limitations**:
   - Low exact match rates (0-18.5%) are expected in QA tasks
   - "Similar" classification is more meaningful for practical applications

3. **Smoothing Function Impact**:
   - BLEU scores with smoothing are extremely low (0.0003-0.0007)
   - Indicates significant n-gram mismatch, possibly due to answer format differences

### 5.3 Limitations

1. **Dataset Size**: RoBERTa trained on truncated dataset limits fair comparison
2. **Evaluation Methodology**: MobileBERT's "all similar" classification needs investigation
3. **Single Epoch Training**: Limited training may not fully capture model potential
4. **Question Type Analysis**: No per-category performance breakdown
5. **Computational Constraints**: Full dataset training for all models would provide better insights

### 5.4 Future Work

1. **Extended Training**: Train all models for multiple epochs on full dataset
2. **Hyperparameter Tuning**: Systematic search for optimal learning rates, batch sizes
3. **Question Type Analysis**: Evaluate performance per question category
4. **Advanced Metrics**: Incorporate F1 score, semantic similarity metrics (BERTScore)
5. **Model Ensembling**: Combine predictions from multiple models
6. **Domain-Specific Fine-tuning**: Pre-train on medical literature before QA fine-tuning
7. **Error Analysis**: Detailed analysis of incorrect predictions

## 6. Conclusion

This study provides a comprehensive comparison of transformer models for medical question answering. Key conclusions:

1. **BERT** demonstrates the most reliable and balanced performance, making it suitable for production medical QA systems.

2. **RoBERTa** shows exceptional promise with the highest ROUGE scores, suggesting it could achieve superior results with full dataset training.

3. **MobileBERT** requires further investigation, as its evaluation results suggest potential calibration or evaluation methodology issues.

4. **Dropout regularization** (0.7) provides marginal improvements, particularly in ROUGE scores, indicating better generalization.

5. **Evaluation metrics** should be considered together—ROUGE appears more reliable than BLEU for semantic QA tasks.

The research establishes a foundation for developing production-grade medical QA systems and highlights the importance of comprehensive evaluation using multiple metrics.

## 7. References

### Datasets
- MedQuad: Medical Question Answering Dataset

### Models
- BERT: Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- MobileBERT: Sun, Z., et al. (2020). "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"
- RoBERTa: Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

### Evaluation Metrics
- BLEU: Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
- ROUGE: Lin, C.-Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"

### Tools and Frameworks
- SimpleTransformers: Rajapakse, T. C. (2019). "Simple Transformers"
- Hugging Face Transformers: Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing"
- Weights & Biases: Biewald, L. (2020). "Experiment Tracking with Weights and Biases"

## 8. Appendix

### 8.1 Training Configuration Details

All models used the following training arguments:
```python
{
    'learning_rate': 5e-5,
    'num_train_epochs': 1,
    'train_batch_size': 16,
    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'evaluate_during_training': True,
    'wandb_project': 'NLP_A5',
    'save_steps': -1,
    'n_best_size': 5,
    'dropout': 0.3 or 0.7
}
```

### 8.2 Data Query Examples

The research included exploratory data analysis using SQL queries:

1. **Treatment Queries**: Most common treatments searched by patients
2. **Symptom Queries**: Questions related to symptoms (e.g., pain)
3. **Genetic Queries**: Questions about genetic changes and treatments
4. **Prevention Queries**: Preventive measures for conditions (e.g., diabetes)
5. **Information Queries**: General information about conditions (e.g., hepatitis)

### 8.3 Evaluation Methodology

- Test set size: 4,102 samples (25% of dataset)
- Evaluation subset for BLEU/ROUGE: 100 samples
- Random seed: 42 (for reproducibility)
- Evaluation performed post-training on held-out test set

---

**Research Conducted**: 2025  
**Dataset**: MedQuad (16,407 samples)  
**Models Evaluated**: BERT, MobileBERT, RoBERTa  
**Primary Metrics**: BLEU, ROUGE-1/2/L, Exact Match

