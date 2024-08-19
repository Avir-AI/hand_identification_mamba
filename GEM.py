import torch.nn.functional as F
import torch
import torch.nn as nn

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, p=3.0):
        super(GeneralizedMeanPooling, self).__init__()
        self.p = p

    def forward(self, x):
        # x is expected to have shape (batch_size, channels, height, width)
        # Flatten the height and width dimensions
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)  # Shape: (batch_size, channels, height * width)
        
        if self.p == 0:
            # Min pooling
            pooled = x.min(dim=-1)[0]
        else:
            # Generalized Mean Pooling
            pooled = torch.mean(x ** self.p, dim=-1) ** (1.0 / self.p)
        
        return pooled