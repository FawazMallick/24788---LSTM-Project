# Turbofan RUL Prediction — 24-788 Deep Learning Mini-Project
 
## Overview
This project compares a supervised GRU baseline and Embed-RUL (Gugulothu et al., 2017)
for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan engine dataset.
Both models are trained on FD001 only and evaluated zero-shot on all four subsets (FD001–FD004).
 
## Repository Contents
- `deep-learning-project.ipynb` — full training and evaluation notebook
- `report.pdf` — project report
## Dependencies
```
torch
numpy
pandas
scikit-learn
matplotlib
scipy
```
Install with:
```
pip install torch numpy pandas scikit-learn matplotlib scipy
```
 
## Data
Download the C-MAPSS dataset from Kaggle (public, no login required):
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Place the files in the same directory as the notebook and update the 
file paths in the **Dataset Paths** cell accordingly.
## Reproducing Results
Run all cells in `deep-learning-project.ipynb` top to bottom. Training takes
approximately 30–60 minutes on a GPU. Key results:
- GRU FD001 RMSE: ~13.2
- Embed-RUL FD001 RMSE: ~64.0
- t-SNE embedding visualization confirms encoder separates healthy vs degraded states
- Prediction figures generated for all four datasets
## Note on Model Checkpoints
Trained model weights (`gru_fd001.pth`, `encoder_fd001.pth`) are not included
due to session expiry during training. Run the notebook top to bottom to reproduce them.
 
## AI Tool Use
Claude (Anthropic) was used to assist with code organization, debugging, and report writing.
All model implementations, training runs, experimental results, and analysis are the author's own work.
 
## Reference
Gugulothu et al., "Predicting Remaining Useful Life using Time Series Embeddings
based on Recurrent Neural Networks," 2nd ML for PHM Workshop at KDD, 2017.
 
