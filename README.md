# Neural Language Model (Assignment 2)

This project trains three neural language models **from scratch** using PyTorch:

- **Underfitting model**
- **Overfitting model**
- **Best-fit model (Transformer)**

Dataset used:
- *Pride and Prejudice* by Jane Austen (provided dataset)

we can run using google collab
!python train_lm.py


All outputs are saved in:

outputs/underfit/
outputs/overfit/
outputs/best_fit/


Each folder contains:
- `loss.png` — training/validation loss curves  
- `results.txt` — test perplexity  
- `best_model.pt` — saved model checkpoint  

## Requirements
- Python 3
- PyTorch
- Matplotlib
