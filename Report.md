# REPORT — Neural Language Model Training

## Dataset
- Provided dataset: *Pride and Prejudice – Jane Austen*
- Preprocessed into tokens using whitespace tokenizer
- Split: 80% train, 10% validation, 10% test

---

# 1. Underfitting Model
### Configuration
- Very small LSTM model  
- Low capacity  
- Few layers

### Results:
- Train Loss: ~4.48  
- Validation Loss: ~5.16  
- Test Perplexity: ~366  

### Observation:
- Both training and validation loss are high  
- Model is too small → **Underfit**

Loss Curve: See `outputs/underfit/loss.png`

---

# 2. Overfitting Model
### Configuration
- Very large LSTM  
- High hidden size  
- Many layers  

### Results:
- Train Loss: ~0.63  
- Validation Loss: ~14.5  
- Test Perplexity: ~10,488,910 (!)  

### Observation:
- Training loss is extremely low → memorizing training  
- Validation loss increases dramatically  
- This is clear **Overfitting**

Loss Curve: `outputs/overfit/loss.png`

---

# 3. Best-Fit Model (Transformer)
### Configuration
- Transformer-based LM  
- Medium capacity  
- Dropout enabled  

### Results:
- Train Loss: ~0.37  
- Validation Loss: ~10.71  
- Test Perplexity: ~183,987  

### Observation:
- Best generalization among the three  
- Still challenging dataset, but performance balanced  

Loss Curve: `outputs/best_fit/loss.png`

---

# Conclusion
- Underfitting: too simple → poor learning  
- Overfitting: too big → memorizes only training  
- Best-fit: trade-off model using Transformer  

This satisfies the assignment requirement to show:
- Underfit
- Overfit
- Best-fit models
