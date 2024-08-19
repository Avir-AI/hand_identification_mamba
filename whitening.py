import torch
import torch.nn as nn

class WhiteningNormalization(nn.Module):
    def __init__(self, num_features):
        super(WhiteningNormalization, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_features)
        batch_size, num_features = x.shape
        
        # Compute the mean and covariance matrix
        mean = x.mean(dim=0)  # Shape: (num_features,)
        centered = x - mean   # Center the data
        
        # Compute covariance matrix
        cov_matrix = torch.matmul(centered.T, centered) / (batch_size - 1)
        
        # Compute the whitening matrix
        u, s, v = torch.svd(cov_matrix)  # Singular Value Decomposition
        whitening_matrix = torch.matmul(v, torch.matmul(torch.diag(1.0 / torch.sqrt(s + 1e-8)), v.T))  # Whitening matrix

        # Apply whitening normalization
        whitened = torch.matmul(centered, whitening_matrix)  # Shape: (batch_size, num_features)
        
        return whitened