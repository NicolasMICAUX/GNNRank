"""Compute `upset_naive` and `upset_simple` on Tournesol "ground truth"."""
import scipy
import numpy as np
import torch
from GNNRank.src.metrics import calculate_upsets


# load data adj.npz and y.npy
A = scipy.sparse.load_npz('adj.npz')
score = np.load('y.npy')

# Convert to FloatTensor
A = torch.from_numpy(A.toarray()).float()
score = torch.from_numpy(score).float()
score = score.unsqueeze(1)


upset_simple = calculate_upsets(A, score, style='simple')
upset_naive = calculate_upsets(A, score, style='naive')
print('upset_simple', upset_simple.item())
print('upset_naive', upset_naive.item())

# upset_simple 0.9663375020027161
# upset_naive 0.2541268765926361