# Code for SLDL.

Code for "Scalable Label Distribution Learning for Multi-Label Classification".

## Requirements

- Python >= 3.6
- PyTorch >= 1.10
- NumPy >= 1.13.3
- Scikit-learn >= 0.20
- Pyxclib >= 0.97

## Running the scripts

To train and test the SLDL model in the terminal, use:

```bash
$ python run_SLDL.py --dataset sample_data --method ridge-lbfgs --label_embedding_threshold 0.1 --label_embedding_dim 128 --knn_neighbors 100 --device cpu --seed 0
```
